"""MLX SimNPO unlearning — reference-free, length-normalised NPO on Apple Silicon (#2005).

The objective is the one the transformers backend runs for ``task: unlearn``
with ``unlearn_method: simnpo`` (``trainer/unlearn.py`` with
:func:`soup_cli.utils.unlearn_kernels.simnpo_loss`):

- ``s``: the summed log-probability of a forget row's target tokens, and ``n``
  the number of them that are scored (every target token after the first
  position of the sequence, as the shifted labels count them);
- forget term ``(2/beta) * mean(softplus(beta * (s/n - gamma)))``, which is
  ``simnpo_loss``'s ``-logsigmoid(-beta * r)``, with ``beta`` the transformers
  path's ``_DEFAULT_BETA`` and ``gamma`` 0;
- with retain rows, ``+ unlearn_alpha * retain CE``, a token-mean
  cross-entropy over the retain rows' target tokens across the batch;
  ``unlearn_alpha`` omitted is 1, and 0 stays 0.

Rows come from ``data.forget_set`` and ``data.retain_set`` through the
transformers path's own loader, and are tokenized as its ``_tokenize_pair``
does: the prompt rendered as one user turn by the tokenizer's chat template
(the raw prompt without one), the target and EOS after it, cut to
``data.max_length`` from the end. A row with no scored token left after that
cut is skipped and counted, never scored against a clamped length.

Each update takes ``batch_size`` forget rows and the next ``batch_size`` retain
rows round-robin. The loop is MLX DPO's, with no reference model, and unlike
the preference objectives it keeps the last partial batch and the last partial
accumulation group, so every forget row trains in every epoch.

Scope (#2005): SimNPO only, on a text Llama-architecture base, with standard
LoRA and ``quantization: none``. NPO and RMU are refused on MLX. The claim is a
lower measured likelihood on the forget rows; nothing here establishes erasure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from rich.console import Console

from soup_cli.trainer.mlx_dpo import MLXDPOTrainerWrapper
from soup_cli.trainer.mlx_orpo import _encode
from soup_cli.trainer.unlearn import _DEFAULT_BETA, _load_unlearn_rows_counted

console = Console()

#: The SimNPO temperature and margin the transformers path trains with.
SIMNPO_BETA = _DEFAULT_BETA
SIMNPO_GAMMA = 0.0
#: The one architecture this path is qualified on (``mlx_lm.models.llama``).
SUPPORTED_MODEL_TYPES = ("llama",)


@dataclass(frozen=True)
class UnlearnExample:
    """One tokenized row: the sequence and where its target starts."""

    ids: list
    target_start: int

    @property
    def scored(self) -> int:
        """Target tokens with a position before them to predict them from."""
        return len(self.ids) - max(1, self.target_start)


def build_unlearn_example(tokenizer: Any, prompt: str, target: str, *,
                          max_length: int) -> UnlearnExample:
    """The ids ``live_eval._tokenize_pair`` builds, as a list, and the target start."""
    from soup_cli.utils.live_eval import _apply_prompt_template

    # The Hugging Face tokenizer under mlx-lm's TokenizerWrapper, which is what
    # the transformers path renders and encodes with.
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    prompt_ids = _encode(hf, _apply_prompt_template(hf, prompt))
    target_ids = _encode(hf, target)
    eos = getattr(hf, "eos_token_id", None)
    if eos is not None:
        target_ids = target_ids + [int(eos)]
    ids = (prompt_ids + target_ids)[:max_length]
    return UnlearnExample(ids=ids, target_start=min(len(prompt_ids), len(ids)))


def collate_unlearn(examples: list) -> tuple[list, list]:
    """``(tokens, target mask)``, right-padded with 0."""
    width = max(len(e.ids) for e in examples)
    tokens = [e.ids + [0] * (width - len(e.ids)) for e in examples]
    mask = [
        [1 if e.target_start <= i < len(e.ids) else 0 for i in range(width)]
        for e in examples
    ]
    return tokens, mask


def simnpo_forget_loss(logps: Any, lengths: Any, *, beta: float, gamma: float) -> Any:
    """``(2/beta) * mean(softplus(beta * (logps/lengths - gamma)))``."""
    import mlx.core as mx

    ratio = logps.astype(mx.float32) / lengths.astype(mx.float32) - gamma
    return (2.0 / beta) * mx.logaddexp(mx.zeros_like(ratio), beta * ratio).mean()


def plan_unlearn_batches(n_rows: int, batch_size: int, epochs: int, seed: int) -> list:
    """Every epoch's batches, shuffled reproducibly from ``seed``, keeping the
    last partial batch (MLX DPO's :func:`plan_batches` drops it)."""
    rng = random.Random(seed)
    plan = []
    for _ in range(max(1, int(epochs))):
        order = list(range(n_rows))
        rng.shuffle(order)
        plan.extend(order[i : i + batch_size] for i in range(0, n_rows, batch_size))
    return plan


def _skip_note(skipped: dict) -> str:
    labels = {
        "blank": "blank", "bad_json": "not JSON", "over_size": "over the size limit",
        "no_pair": "no usable prompt/target",
    }
    parts = [f"{skipped[k]} {label}" for k, label in labels.items() if skipped.get(k)]
    if skipped.get("past_row_limit"):
        parts.append("lines past the row limit not read")
    return ", ".join(parts)


class MLXSimNPOTrainerWrapper(MLXDPOTrainerWrapper):
    """SimNPO on MLX: the transformers path's rows and loss, MLX DPO's loop."""

    objective = "SimNPO"
    metric_names: tuple = ("forget_loss", "retain_ce", "forget_mean_logp")
    flush_partial_accumulation = True
    reference_unit = "row(s)"

    # -- setup ------------------------------------------------------------

    def setup(self, dataset: Any = None) -> None:
        """Read the forget and retain rows, then load the model.

        ``dataset`` (``data.train``) is not read, as on the transformers path:
        the rows are ``data.forget_set`` and ``data.retain_set``.
        """
        self._require_mlx()
        self._check_unsupported()
        self._load_rows()

        from soup_cli.utils.mlx import load_mlx_model

        cfg = self.config
        console.print(f"[dim]Loading MLX model for SimNPO: {cfg.base}[/]")
        self.model, self.tokenizer = load_mlx_model(
            cfg.base, quantization=cfg.training.quantization
        )
        self._check_architecture(self.model)
        console.print(
            f"[green]MLX model loaded:[/] {cfg.base} "
            f"(task=unlearn, unlearn_method=simnpo, lora_r={cfg.training.lora.r})"
        )

    def _load_rows(self) -> None:
        """Both row sets, before any model loads; refuses an unusable one."""
        data = self.config.data
        self._forget_rows, forget_skipped = _load_unlearn_rows_counted(data.forget_set)
        self._report_skipped("forget_set", data.forget_set, forget_skipped)
        if not self._forget_rows:
            raise ValueError(f"forget_set {data.forget_set!r} yielded no usable rows")
        self._retain_rows: list = []
        if data.retain_set:
            self._retain_rows, retain_skipped = _load_unlearn_rows_counted(data.retain_set)
            self._report_skipped("retain_set", data.retain_set, retain_skipped)
            if not self._retain_rows:
                raise ValueError(f"retain_set {data.retain_set!r} yielded no usable rows")
        else:
            # The transformers path's warning, word for word.
            console.print(
                "[yellow]Unlearn warning:[/] method=simnpo has no "
                "retain_set; general capability is unanchored. Supply "
                "data.retain_set to preserve utility."
            )
            self.metric_names = ("forget_loss", "forget_mean_logp")
        self._dataset = {"train": self._forget_rows}

    @staticmethod
    def _report_skipped(field: str, path: str, skipped: dict) -> None:
        note = _skip_note(skipped)
        if note:
            console.print(f"[yellow]MLX SimNPO: {field} {path!r}: skipped {note}[/]")

    @staticmethod
    def _check_architecture(model: Any) -> None:
        model_type = getattr(model, "model_type", None)
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"MLX SimNPO is qualified on Llama-architecture bases only "
                f"(model_type 'llama'); this base is {model_type!r}. Use "
                "backend: transformers for it."
            )

    # -- data -------------------------------------------------------------

    def _tokenize(self, rows: list, field: str) -> list:
        """Tokenized rows with a scored token; the rest are counted and reported."""
        max_length = int(getattr(self.config.data, "max_length", 2048) or 2048)
        examples, empty = [], 0
        for prompt, target in rows:
            example = build_unlearn_example(
                self.tokenizer, prompt, target, max_length=max_length
            )
            if example.scored > 0:
                examples.append(example)
            else:
                empty += 1
        if empty:
            console.print(
                f"[yellow]MLX SimNPO: skipped {empty} {field} row(s) with no target "
                f"token left after the cut to data.max_length ({max_length})[/]"
            )
        self._empty_rows[field] = empty
        return examples

    def _examples(self, rows: list, split: str = "train") -> tuple[list, int]:
        if split != "train":
            return [], 0
        self._empty_rows: dict = {}
        examples = self._tokenize(rows, "forget_set")
        self._retain_examples = self._tokenize(self._retain_rows, "retain_set")
        if self._retain_rows and not self._retain_examples:
            raise ValueError(
                f"MLX SimNPO: no retain_set row has a target token left after the cut "
                f"to data.max_length; {len(self._retain_rows)} row(s) read"
            )
        self._retain_cursor = 0
        self._scored = {"forget": 0, "retain": 0}
        return examples, 0

    def _check_train_rows(self, n_rows: int, batch_size: int, ties: int) -> None:
        if n_rows == 0:
            raise ValueError(
                "MLX SimNPO: no forget_set row has a target token left after the cut "
                f"to data.max_length; {len(self._forget_rows)} row(s) read"
            )

    def _plan_batches(self, n_rows: int, batch_size: int, epochs: int, seed: int) -> list:
        return plan_unlearn_batches(n_rows, batch_size, epochs, seed)

    @staticmethod
    def _arrays(examples: list) -> tuple:
        import mlx.core as mx

        tokens, mask = collate_unlearn(examples)
        return mx.array(tokens), mx.array(mask)

    def _verification_arrays(self, examples: list) -> tuple:
        return self._arrays(examples)

    def _batch_arrays(self, examples: list) -> tuple:
        """``(tokens, mask)`` of the forget batch, then of as many retain rows,
        taken round-robin, when there is a retain set."""
        forget = self._arrays(examples)
        self._scored["forget"] += len(examples)
        if not self._retain_examples:
            return forget
        pool = self._retain_examples
        picked = [
            pool[(self._retain_cursor + i) % len(pool)] for i in range(len(examples))
        ]
        self._retain_cursor = (self._retain_cursor + len(examples)) % len(pool)
        self._scored["retain"] += len(picked)
        return forget + self._arrays(picked)

    # -- objective --------------------------------------------------------

    def _prepare_objective(self, train_examples: list, val_examples: list,
                           batch_size: int) -> None:
        """SimNPO has no reference model."""

    def _alpha(self) -> float:
        alpha = self.config.training.unlearn_alpha
        return 1.0 if alpha is None else float(alpha)

    def _loss(self, model: Any, arrays: tuple, split: str, rows: Any) -> tuple:
        from soup_cli.trainer.mlx_dpo import sequence_logps

        tokens, mask = arrays[:2]
        lengths = mask[:, 1:].sum(axis=-1)
        logps = sequence_logps(model, tokens, mask, head_split=self._head_split)
        forget = simnpo_forget_loss(logps, lengths, beta=SIMNPO_BETA, gamma=SIMNPO_GAMMA)
        metrics = {
            "forget_loss": forget,
            "forget_mean_logp": (logps / lengths).mean(),
        }
        if len(arrays) == 2:
            return forget, metrics
        r_tokens, r_mask = arrays[2:]
        r_logps = sequence_logps(model, r_tokens, r_mask, head_split=self._head_split)
        retain_ce = -r_logps.sum() / r_mask[:, 1:].sum()
        metrics["retain_ce"] = retain_ce
        return forget + self._alpha() * retain_ce, metrics

    def _format_metrics(self, metrics: dict) -> str:
        text = f"forget {metrics['forget_loss']:.4f}"
        if "retain_ce" in metrics:
            text += f", retain CE {metrics['retain_ce']:.4f}"
        return text

    def _objective_metadata(self) -> dict:
        return {
            "task": "unlearn",
            "unlearn_method": "simnpo",
            "simnpo_beta": SIMNPO_BETA,
            "simnpo_gamma": SIMNPO_GAMMA,
            "unlearn_alpha": self._alpha() if self._retain_rows else None,
            "forget_rows": len(self._forget_rows),
            "retain_rows": len(self._retain_rows),
            "simnpo_loss": "(2/beta)*mean(softplus(beta*(s/n-gamma))) + alpha*retain token-mean CE",
            "reference": "none (reference-free)",
        }

    # -- loop -------------------------------------------------------------

    def train(self, display=None, tracker=None, run_id=None,
              resume_from_checkpoint=None) -> dict:
        result = super().train(
            display=display, tracker=tracker, run_id=run_id,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        result.pop("skipped_identical_pairs", None)
        result.update({
            "updates": len(result.get("losses", [])),
            "forget_rows_scored": self._scored["forget"],
            "retain_rows_scored": self._scored["retain"],
            "forget_rows_without_target": self._empty_rows.get("forget_set", 0),
            "retain_rows_without_target": self._empty_rows.get("retain_set", 0),
        })
        console.print(
            f"[dim]MLX SimNPO: {result['updates']} update(s), "
            f"{result['forget_rows_scored']} forget and "
            f"{result['retain_rows_scored']} retain row(s) scored[/]"
        )
        return result
