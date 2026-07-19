"""HuggingFace auto-push callback and resume helpers (v0.29.0 Part B).

Hooks into the HF Trainer's ``on_save`` event so every checkpoint saved to
disk is pushed to the Hub as a ``checkpoint-<step>`` branch. Also provides
``prepare_hf_resume`` which downloads the latest checkpoint branch back to
the local ``output_dir`` so a fresh run can pick up where the previous
crashed.

Network errors are logged and swallowed — we never crash training because
the Hub is unreachable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from soup_cli.utils.hf import get_hf_api, resolve_endpoint, resolve_token, validate_repo_id

logger = logging.getLogger(__name__)

_CHECKPOINT_BRANCH_RE = re.compile(r"^checkpoint-(\d+)$")

# Files worth shipping in an auto-pushed checkpoint. Keeps stray .env /
# source files / caches out of auto-pushed revisions if ``output_dir``
# is ever misconfigured to overlap with the project root.
_CHECKPOINT_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.json",
    "tokenizer*",
    "special_tokens_map.json",
    "generation_config.json",
    "trainer_state.json",
    "training_args.bin",
    "README.md",
]


def _try_import_callback_base():
    """Return HF ``TrainerCallback`` (or ``object`` when transformers is absent).

    Imported inside the function so the module has no top-level transformers
    dependency; the class below still inherits every no-op event stub the HF
    dispatch loop requires. Mirrors ``monitoring/curriculum_callback.py`` /
    ``utils/lisa.py``.
    """
    try:
        from transformers import TrainerCallback  # noqa: PLC0415

        return TrainerCallback
    except Exception:  # noqa: BLE001 — transformers optional in slim test envs.
        return object


class HFPushCallback(_try_import_callback_base()):  # type: ignore[misc]
    """Real HF ``TrainerCallback`` auto-pusher.

    Subclasses the lazily-resolved ``TrainerCallback`` so it inherits the no-op
    default for every Trainer event — HF's ``CallbackHandler.call_event``
    dispatches every event via ``getattr(cb, event)`` with no ``hasattr`` guard,
    so a bare duck-typed callback crashes on ``on_epoch_begin`` (#308). Only
    ``on_train_begin`` / ``on_save`` are overridden. The lazy factory keeps the
    module import free of transformers.
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        endpoint: Optional[str] = None,
        output_dir: str = "",
        private: bool = False,
    ) -> None:
        validate_repo_id(repo_id)
        self.repo_id = repo_id
        self.token = token
        self.endpoint = endpoint
        # ``output_dir`` is a fallback used only when HF Trainer's
        # ``TrainingArguments.output_dir`` is missing (e.g. tests that
        # construct a bare SimpleNamespace). Under real training the value
        # comes from ``args.output_dir``.
        self.output_dir = output_dir
        self.private = private
        self._repo_created = False
        self._repo_failed = False  # short-circuits retries after hard failure

    # --- TrainerCallback protocol ---

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        # Eagerly create the repo so the first checkpoint upload is not
        # delayed. Swallow errors so we don't crash training; the failure
        # is retried once on the first on_save, then short-circuited.
        self._ensure_repo()

    def on_save(self, args, state, control, **kwargs) -> None:
        """Upload the checkpoint directory written at ``global_step``."""
        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0:
            return

        out_dir = getattr(args, "output_dir", None) or self.output_dir
        if not out_dir:
            return

        ckpt_path = Path(out_dir) / f"checkpoint-{step}"
        if not ckpt_path.is_dir():
            logger.debug("HFPushCallback: checkpoint dir missing: %s", ckpt_path)
            return

        self._upload_checkpoint(ckpt_path, step)

    # --- Helpers ---

    def _ensure_repo(self) -> bool:
        """Create the repo if needed. Returns True if ready for uploads."""
        if self._repo_created:
            return True
        if self._repo_failed:
            return False
        try:
            api = get_hf_api(token=self.token, endpoint=self.endpoint)
            api.create_repo(repo_id=self.repo_id, private=self.private, exist_ok=True)
            self._repo_created = True
            return True
        except Exception as exc:
            logger.warning(
                "HFPushCallback: create_repo failed (%s); auto-push disabled", exc,
            )
            self._repo_failed = True
            return False

    def _upload_checkpoint(self, ckpt_path: Path, step: int) -> None:
        if not self._ensure_repo():
            return
        try:
            api = get_hf_api(token=self.token, endpoint=self.endpoint)

            revision = f"checkpoint-{step}"
            try:
                api.create_branch(
                    repo_id=self.repo_id,
                    branch=revision,
                    exist_ok=True,
                )
            except Exception as exc:
                # create_branch is best-effort — older hub versions lack it.
                logger.debug("create_branch failed (continuing): %s", exc)

            api.upload_folder(
                folder_path=str(ckpt_path),
                repo_id=self.repo_id,
                revision=revision,
                commit_message=f"Soup auto-push checkpoint-{step}",
                allow_patterns=_CHECKPOINT_ALLOW_PATTERNS,
            )
        except Exception as exc:
            logger.warning("HFPushCallback: upload failed at step %d: %s", step, exc)


def resolve_latest_checkpoint_revision(
    repo_id: str,
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> Optional[str]:
    """Return the ``checkpoint-<N>`` revision with the largest ``N``, or None.

    Swallows API errors — a missing repo or network failure returns None so
    callers can fall back to training from scratch.
    """
    try:
        api = get_hf_api(token=token, endpoint=endpoint)
        refs = api.list_repo_refs(repo_id=repo_id)
    except Exception as exc:
        logger.debug("list_repo_refs failed (%s); no resume revision", exc)
        return None

    branches = getattr(refs, "branches", None) or []
    best_step = -1
    best_name: Optional[str] = None
    for branch in branches:
        name = getattr(branch, "name", None)
        if not isinstance(name, str):
            continue
        match = _CHECKPOINT_BRANCH_RE.match(name)
        if not match:
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_name = name
    return best_name


def _find_highest_local_checkpoint(output_dir: str) -> Optional[int]:
    """Return the highest ``checkpoint-<N>`` step under ``output_dir``, or None.

    Skips non-directories and malformed names. Returns None if ``output_dir``
    does not exist or contains no checkpoints.
    """
    base = Path(output_dir)
    if not base.is_dir():
        return None
    best: Optional[int] = None
    try:
        children = list(base.iterdir())
    except OSError:
        return None
    for entry in children:
        if not entry.is_dir():
            continue
        match = _CHECKPOINT_BRANCH_RE.match(entry.name)
        if not match:
            continue
        step = int(match.group(1))
        if best is None or step > best:
            best = step
    return best


def _download_checkpoint(
    repo_id: str,
    revision: str,
    local_dir: str,
    token: Optional[str],
    endpoint: Optional[str],
) -> str:
    """Download a revision into ``local_dir``.

    ``local_dir_use_symlinks=False`` forces direct copies — defence against
    older ``huggingface_hub`` versions that could symlink the shared cache
    into ``local_dir`` and thus let a crafted repo (or future SDK bug) place
    symlinks pointing at arbitrary filesystem locations.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for --hf-resume. Install huggingface-hub."
        ) from exc

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    try:
        resolved = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            token=token,
            endpoint=endpoint,
            local_dir_use_symlinks=False,
        )
    except TypeError:
        # Older huggingface_hub versions reject local_dir_use_symlinks.
        resolved = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            token=token,
            endpoint=endpoint,
        )
    return resolved or local_dir


def prepare_hf_resume(
    repo_id: str,
    output_dir: str,
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> Optional[str]:
    """Pull the latest checkpoint branch from HF into ``output_dir``.

    Returns the local path of the checkpoint directory, or None if there's
    nothing to resume from.

    The ``output_dir`` must stay under the current working directory — an
    attacker-controlled ``cfg.output`` (e.g. ``../../../tmp``) would
    otherwise place downloaded files outside the project tree.
    """
    from soup_cli.utils.paths import is_under_cwd

    validate_repo_id(repo_id)
    if not is_under_cwd(output_dir):
        raise ValueError(
            "output_dir must stay under the current working directory "
            f"for --hf-resume; got: {output_dir!r}"
        )
    revision = resolve_latest_checkpoint_revision(repo_id, token=token, endpoint=endpoint)
    if revision is None:
        return None

    # Prefer local newer (#50): if local has checkpoint-N >= remote's,
    # skip the download and return the local path. Saves bandwidth and
    # avoids overwriting a fresher local checkpoint with stale Hub state.
    remote_match = _CHECKPOINT_BRANCH_RE.match(revision)
    remote_step = int(remote_match.group(1)) if remote_match else -1
    local_step = _find_highest_local_checkpoint(output_dir)
    if local_step is not None and local_step >= remote_step:
        local_revision = f"checkpoint-{local_step}"
        local_path = Path(output_dir) / local_revision
        logger.info(
            "HF resume: local %s >= remote %s; skipping download",
            local_revision,
            revision,
        )
        return str(local_path)

    # Mirror HF Trainer's on-disk layout: output_dir/<revision>
    local_dir = str(Path(output_dir) / revision)
    try:
        return _download_checkpoint(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            token=token,
            endpoint=endpoint,
        )
    except Exception as exc:
        logger.warning("HF resume download failed (%s); skipping auto-resume", exc)
        return None


def build_push_callback(
    repo_id: str,
    output_dir: str,
    explicit_token: Optional[str] = None,
    private: bool = False,
) -> Optional[HFPushCallback]:
    """Factory that resolves token/endpoint and builds the callback.

    Returns None when no HF token is available — the caller logs and skips
    auto-push silently.
    """
    token = resolve_token(explicit=explicit_token)
    if token is None:
        return None

    try:
        endpoint = resolve_endpoint()
    except ValueError as exc:
        logger.warning("HF_ENDPOINT invalid (%s); skipping auto-push", exc)
        return None

    return HFPushCallback(
        repo_id=repo_id,
        token=token,
        endpoint=endpoint,
        output_dir=output_dir,
        private=private,
    )
