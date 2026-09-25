"""Online DPO pairs the judge could not rank (#1225).

Soup's pairwise judge (:func:`soup_cli.eval.judge.make_soup_pairwise_judge`)
answers ``-1`` when it cannot name a winner: a tie, a failed or unreadable
judge call, or a verdict that changes when the two completions are swapped.
trl 0.29's ``OnlineDPOTrainer`` does not drop a ``-1``. Its ``training_step``
builds ``mask = rank == 0``, so every such pair trained as "the second
completion wins", and it averages the loss over every pair. With the judge
server down, a whole run learned arbitrary labels and finished normally.

:func:`make_ranked_pairs_trainer` keeps trl's step and works around it rather
than copying it: it records the ranks the judge returns, gives the pairs it
could not rank exactly zero gradient through a hook on the policy log-probs,
reports the loss and the chosen/rejected statistics of the ranked pairs only,
logs the unranked share as ``judge/invalid_rate``, and stops the run when the
judge has ranked nothing for :data:`MAX_CONSECUTIVE_UNRANKED_PAIRS` pairs.
:func:`report_unranked_pairs` closes the gap for a run shorter than that: a
run in which the judge ranked no pair at all fails instead of saving an
adapter as if it had trained.

A micro-batch in which nothing was ranked adds no gradient, but it is not
skipped: the optimizer still steps, so AdamW momentum and weight decay keep
moving the weights and the learning-rate schedule advances. Skipping
``optimizer.step`` from inside the Trainer is fragile, especially under
DeepSpeed, whose engine steps inside ``backward``. The drift is bounded by the
stop after :data:`MAX_CONSECUTIVE_UNRANKED_PAIRS` unranked pairs.

Import-light: torch is imported inside the functions that need it.
"""

import logging
import re
from functools import lru_cache
from typing import Optional

from soup_cli.utils.terminal import strip_control

logger = logging.getLogger(__name__)

# The ``user:password@`` part of a URL: everything between ``//`` and an ``@``
# with no ``/`` in between (an ``@`` in the path is not userinfo).
_USERINFO_RE = re.compile(r"(//)[^/@]*@")

# Training stops once this many pairs in a row, counted over consecutive steps
# in which the judge ranked nothing, could not be ranked. Counted in pairs
# rather than steps so the stop does not depend on the batch size: at
# batch_size 1, a judge that fails to rank 35% of pairs would lose 8 steps in a
# row about once in 4,400 steps, while 0.35 ** 32 is about 3e-15.
MAX_CONSECUTIVE_UNRANKED_PAIRS = 32
INVALID_RATE_KEY = "judge/invalid_rate"

# trl statistics that depend on which completion was chosen.
_PAIR_STATS = (
    "logps/chosen",
    "logps/rejected",
    "rewards/chosen",
    "rewards/rejected",
    "rewards/margins",
    "rewards/accuracies",
)
_UNRANKED_REASONS = (
    "a tie, a failed or unreadable judge call, or a verdict that changed when "
    "the two completions were swapped"
)


class JudgeUnusableError(RuntimeError):
    """The pairwise judge ranked none of the recent completion pairs (#1225)."""


def _unusable_message(label: str, what: str) -> str:
    return (
        f"Online DPO stopped: the judge {label} {what} ({_UNRANKED_REASONS}), so "
        "there was nothing to train on. Check that training.online_dpo_judge points "
        'at a reachable judge that answers {"winner": "A"} or {"winner": "B"} and '
        "keeps its verdict when the two completions are swapped."
    )


def report_unranked_pairs(trainer) -> Optional[str]:
    """End-of-run account of the pairs the judge could not rank (#1225).

    Raises :class:`JudgeUnusableError` when the judge ranked no pair in the
    whole run, which a run shorter than :data:`MAX_CONSECUTIVE_UNRANKED_PAIRS`
    pairs never reaches mid-run: nothing was trained, so the run must not
    finish as a success. Returns a one-line summary when some pairs were
    unranked, and None otherwise or for a trainer that does not count pairs.
    """
    judged = getattr(trainer, "_soup_judged_pairs", 0)
    unranked = getattr(trainer, "_soup_unranked_pairs", 0)
    if not (isinstance(judged, int) and isinstance(unranked, int)) or not unranked:
        return None
    if unranked == judged:
        label = getattr(trainer, "_soup_judge_label", "the configured judge")
        raise JudgeUnusableError(
            _unusable_message(label, f"ranked none of the {judged} completion pairs of the run")
        )
    return (
        f"the judge could not rank {unranked} of {judged} pairs "
        f"({100 * unranked / judged:.1f}%); they were left out of the loss."
    )


def judge_label(url: Optional[str]) -> str:
    """The judge URL for a message: printable, on one line, credentials masked.

    ``training.online_dpo_judge`` comes from a shareable ``soup.yaml`` and ends
    up in a WARNING and in :class:`JudgeUnusableError`, so control bytes (an
    ESC would start a terminal escape sequence) are stripped with the shared
    :func:`soup_cli.utils.terminal.strip_control`, and line breaks are
    collapsed so the URL cannot forge a log line of its own.
    """
    url = " ".join(strip_control(url or "").split())
    if not url:
        return "the configured judge"
    # a pattern rather than urlparse, which raises on a malformed netloc
    return _USERINFO_RE.sub(r"\1***@", url, count=1)


def _rank_equals(rank, value: int) -> bool:
    """``rank == value`` as trl evaluates it, and False where that cannot be read."""
    try:
        return bool(rank == value)
    except (TypeError, ValueError, RuntimeError):
        return False


def _is_ranked(rank) -> bool:
    """True only for a verdict: 0 (first completion) or 1 (second completion)."""
    return _rank_equals(rank, 0) or _rank_equals(rank, 1)


class _PairStep:
    """What one Online DPO step's judge and forward passes produced."""

    def __init__(self) -> None:
        self.ranks: Optional[list] = None
        self.keep: Optional[list[bool]] = None
        self.policy = None
        self.ref = None
        self.completion_mask = None
        self.hook_ran = False

    def record(self, ranks, n_pairs: int) -> list:
        ranks = list(ranks)
        if len(ranks) != n_pairs:
            raise ValueError(
                f"The Online DPO judge returned {len(ranks)} ranks for {n_pairs} "
                "completion pairs; it must return exactly one per pair (#1225)."
            )
        self.ranks = ranks
        self.keep = [_is_ranked(rank) for rank in ranks]
        return ranks

    def observe(self, logprobs, completion_mask) -> None:
        """Keep the policy (with grad) and reference (no grad) per-token log-probs."""
        if logprobs.requires_grad:
            if self.policy is None:
                self.policy = logprobs.detach()
                self.completion_mask = completion_mask
                logprobs.register_hook(self._reweight)
        elif self.ref is None:
            self.ref = logprobs.detach()

    def _reweight(self, grad):
        """Zero the rows of unranked pairs; average the rest over the ranked pairs.

        trl's loss is the mean over all ``B`` pairs, and pair ``i`` reaches the
        parameters only through rows ``i`` and ``i + B`` of the policy
        log-probs. Scaling the ranked rows by ``B / kept`` and zeroing the others
        gives exactly the gradient of the mean over the ranked pairs.
        """
        self.hook_ran = True
        if self.keep is None or all(self.keep):
            return None
        import torch

        kept = sum(self.keep)
        if kept == 0:
            return torch.zeros_like(grad)
        rows = torch.tensor(self.keep + self.keep, device=grad.device).unsqueeze(1)
        return torch.where(rows, grad * (len(self.keep) / kept), torch.zeros_like(grad))

    def require_complete(self) -> None:
        if self.ranks is None or self.policy is None or self.ref is None or not self.hook_ran:
            raise RuntimeError(
                "Online DPO cannot tell the pairs its judge ranked from the ones it could "
                "not (#1225): the installed trl did not call trainer.judge.judge() and "
                "trainer._forward() the way trl 0.29 does, so an unranked pair could not "
                "be kept out of the gradient. The step was stopped before the optimizer "
                "applied it. Install trl 0.29.x."
            )


class _RecordingJudge:
    """Stands in for ``trainer.judge`` during one step and records its ranks."""

    def __init__(self, judge, step: _PairStep) -> None:
        self._judge = judge
        self._step = step

    def judge(self, prompts, completions, *args, **kwargs):
        ranks = self._judge.judge(prompts, completions, *args, **kwargs)
        return self._step.record(ranks, len(completions))


def _pair_terms(step: _PairStep, beta: float, loss_type: str):
    """Per-pair loss and chosen/rejected sums, computed as trl 0.29 computes them."""
    import torch
    import torch.nn.functional as F  # noqa: N812 - the name trl uses

    n_pairs = len(step.ranks)
    first_wins = torch.tensor(
        [_rank_equals(rank, 0) for rank in step.ranks], device=step.policy.device
    )
    mask = step.completion_mask.bool()
    first, second = (step.policy * mask).sum(1).split(n_pairs)
    ref_first, ref_second = (step.ref * mask).sum(1).split(n_pairs)
    chosen = torch.where(first_wins, first, second)
    rejected = torch.where(first_wins, second, first)
    ref_chosen = torch.where(first_wins, ref_first, ref_second)
    ref_rejected = torch.where(first_wins, ref_second, ref_first)
    logits = (chosen - rejected) - (ref_chosen - ref_rejected)
    if loss_type == "sigmoid":
        losses = -F.logsigmoid(beta * logits)
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    else:
        raise NotImplementedError(f"invalid loss type {loss_type}")
    return losses, torch.stack(
        [chosen, rejected, beta * (chosen - ref_chosen), beta * (rejected - ref_rejected)],
        dim=1,
    )


def _ranked_pair_stats(rows) -> dict:
    """trl's chosen/rejected statistics over ``rows`` of ranked pairs only."""
    chosen, rejected, chosen_rewards, rejected_rewards = rows.unbind(1)
    margins = chosen_rewards - rejected_rewards
    return {
        "logps/chosen": chosen.mean().item(),
        "logps/rejected": rejected.mean().item(),
        "rewards/chosen": chosen_rewards.mean().item(),
        "rewards/rejected": rejected_rewards.mean().item(),
        "rewards/margins": margins.mean().item(),
        "rewards/accuracies": (margins > 0).float().mean().item(),
    }


class _LossAccount:
    """Micro-batches seen, those with a ranked pair, and the sum of their losses.

    A micro-batch is one process's batch in one ``training_step``; its loss is
    the mean over its ranked pairs, the value trl's step optimises.
    """

    def __init__(self) -> None:
        self.units = 0
        self.ranked_units = 0
        self.loss_sum = 0.0

    def add(self, keep, losses, pairs_per_unit: int) -> None:
        """Add one gathered step: ``keep`` and ``losses`` hold one row per pair."""
        import torch

        units = keep.numel() // pairs_per_unit
        keep = keep[: units * pairs_per_unit].view(units, pairs_per_unit) > 0.5
        losses = losses[: units * pairs_per_unit].view(units, pairs_per_unit)
        kept = keep.sum(1)
        ranked = kept > 0
        # torch.where, not a product: an unranked pair's loss may be inf or NaN
        sums = torch.where(keep, losses, torch.zeros_like(losses)).sum(1)
        self.units += units
        self.ranked_units += int(ranked.sum())
        self.loss_sum += float((sums[ranked] / kept[ranked]).sum())

    def fix(self, logs: dict, key: str, digits: Optional[int] = None) -> dict:
        """``logs`` with ``key`` over the ranked micro-batches only.

        The Trainer averages what every ``training_step`` returned, and a
        micro-batch with no ranked pair returns 0. When every micro-batch had a
        ranked pair there is nothing to correct and ``logs`` is returned
        untouched; when none had, ``key`` is dropped rather than reported as a
        loss nothing was trained on.
        """
        if self.ranked_units == self.units:
            return logs
        fixed = dict(logs)
        if not self.ranked_units:
            del fixed[key]
            return fixed
        mean = self.loss_sum / self.ranked_units
        fixed[key] = round(mean, digits) if digits is not None else mean
        return fixed


def make_ranked_pairs_trainer(base_cls: type) -> type:
    """An ``OnlineDPOTrainer`` subclass that trains only the pairs its judge ranked.

    The subclass keeps trl's ``training_step`` and changes three things
    around it:

    * a gradient hook on the policy log-probs gives unranked pairs exactly zero
      gradient and makes the rest a mean over the ranked pairs of the batch; a
      batch with no ranked pair adds no gradient (the optimizer still steps,
      see the module docstring). Under gradient accumulation each micro-batch
      is averaged over its own ranked pairs, which is what trl would do if the
      unranked pairs had never been generated;
    * the returned loss and trl's chosen/rejected statistics describe the
      ranked pairs only; the logged ``loss`` and ``train_loss`` average the
      micro-batches that ranked a pair, and are left out when none did;
      ``judge/invalid_rate`` logs the unranked share;
    * a WARNING the first time a pair is unranked and on every step in which
      nothing was ranked, and :class:`JudgeUnusableError` once
      :data:`MAX_CONSECUTIVE_UNRANKED_PAIRS` pairs in a row were unranked.

    Every process gathers the same tensor whatever its judge returned, so a
    distributed run issues the same collectives on every rank. With clean 0/1
    ranks the step is trl's own, bit for bit. Cached per base class, like
    :func:`soup_cli.trainer.grpo.make_grpo_trainer_variant`.
    """
    return _make_ranked_pairs_trainer_cached(base_cls)


@lru_cache(maxsize=4)
def _make_ranked_pairs_trainer_cached(base_cls: type) -> type:
    class _RankedPairsOnlineDPOTrainer(base_cls):  # type: ignore[misc, valid-type]
        """``OnlineDPOTrainer`` that leaves unranked judge pairs out (#1225)."""

        _soup_judge_label = "the configured judge"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._soup_step: Optional[_PairStep] = None
            self._soup_judged_pairs = 0
            self._soup_unranked_pairs = 0
            self._soup_unranked_streak = 0
            self._soup_warned_partial = False
            # losses of the micro-batches since the last log, and over the run
            self._soup_window = _LossAccount()
            self._soup_run = _LossAccount()

        def training_step(self, model, inputs, num_items_in_batch=None):
            judge = self.judge
            if judge is None:
                return super().training_step(model, inputs, num_items_in_batch)
            step = _PairStep()
            stat_sizes = {key: len(values) for key, values in self.stats.items()}
            self._soup_step = step
            self.judge = _RecordingJudge(judge, step)
            try:
                loss = super().training_step(model, inputs, num_items_in_batch)
            finally:
                self.judge = judge
                self._soup_step = None
            return self._soup_settle(step, loss, stat_sizes)

        def _forward(
            self, model, prompt_ids, prompt_mask, completion_ids, completion_mask,
            vision_inputs=None,
        ):
            logprobs = super()._forward(
                model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs
            )
            step = getattr(self, "_soup_step", None)
            if step is not None:
                step.observe(logprobs, completion_mask)
            return logprobs

        def _maybe_log_save_evaluate(self, *args, **kwargs):
            # A step in which nothing was ranked leaves no chosen/rejected
            # statistics; trl divides by the number of entries when it logs.
            empty = [key for key, values in self.stats.items() if not values]
            for key in empty:
                del self.stats[key]
            try:
                return super()._maybe_log_save_evaluate(*args, **kwargs)
            finally:
                for key in empty:
                    self.stats.setdefault(key, [])

        def log(self, logs, *args, **kwargs):
            # A micro-batch with no ranked pair returns a loss of 0, which the
            # Trainer would average into `loss` and `train_loss`, and a 0.0
            # final loss would win a sweep. Both are reported over the
            # micro-batches that ranked a pair instead, and left out when none did.
            if "loss" in logs:
                logs = self._soup_window.fix(logs, "loss", digits=4)
                self._soup_window = _LossAccount()
            if "train_loss" in logs:
                logs = self._soup_run.fix(logs, "train_loss")
            return super().log(logs, *args, **kwargs)

        def _soup_settle(self, step: _PairStep, loss, stat_sizes: dict):
            import torch

            step.require_complete()
            losses, pair_rows = _pair_terms(step, self.beta, self.args.loss_type)
            keep = torch.tensor(step.keep, dtype=torch.float32, device=pair_rows.device)
            gathered = self.accelerator.gather(
                torch.cat(
                    [keep.unsqueeze(1), losses.float().unsqueeze(1), pair_rows.float()], dim=1
                )
            )
            for account in (self._soup_window, self._soup_run):
                account.add(gathered[:, 0], gathered[:, 1], len(step.keep))
            ranked_rows = gathered[gathered[:, 0] > 0.5, 2:]
            judged, ranked = int(gathered.shape[0]), int(ranked_rows.shape[0])
            self.stats.setdefault(INVALID_RATE_KEY, []).append((judged - ranked) / judged)
            self._soup_judged_pairs += judged
            self._soup_unranked_pairs += judged - ranked
            if ranked < judged:
                self._soup_replace_pair_stats(ranked_rows, stat_sizes)
            self._soup_watch_the_judge(judged, ranked)

            kept = sum(step.keep)
            if kept == len(step.keep):
                return loss
            if kept == 0:
                return torch.zeros_like(loss)
            local_keep = keep.bool().to(losses.device)
            return losses[local_keep].mean() / self.args.gradient_accumulation_steps

        def _soup_replace_pair_stats(self, ranked_rows, stat_sizes: dict) -> None:
            fixed = _ranked_pair_stats(ranked_rows) if ranked_rows.shape[0] else None
            for key in _PAIR_STATS:
                values = self.stats.get(key)
                if values is None or len(values) != stat_sizes.get(key, 0) + 1:
                    continue  # trl did not record this statistic for the step
                if fixed is None:
                    values.pop()
                else:
                    values[-1] = fixed[key]

        def _soup_watch_the_judge(self, judged: int, ranked: int) -> None:
            unranked = judged - ranked
            if ranked:
                self._soup_unranked_streak = 0
                if unranked and not self._soup_warned_partial:
                    self._soup_warned_partial = True
                    if self.accelerator.is_main_process:
                        logger.warning(
                            "Online DPO: the judge %s could not rank %d of %d pairs in this "
                            "step (%s). Those pairs are left out of the loss; their share "
                            "is logged as %s.",
                            self._soup_judge_label, unranked, judged, _UNRANKED_REASONS,
                            INVALID_RATE_KEY,
                        )
                return
            self._soup_unranked_streak += judged
            if self._soup_unranked_streak >= MAX_CONSECUTIVE_UNRANKED_PAIRS:
                raise JudgeUnusableError(
                    _unusable_message(
                        self._soup_judge_label,
                        f"ranked none of the last {self._soup_unranked_streak} "
                        "completion pairs",
                    )
                )
            if self.accelerator.is_main_process:
                logger.warning(
                    "Online DPO: the judge %s could not rank any of the %d pairs in this "
                    "step (%s); the step adds no gradient. %d pairs in a row are unranked; "
                    "training stops at %d.",
                    self._soup_judge_label, judged, _UNRANKED_REASONS,
                    self._soup_unranked_streak, MAX_CONSECUTIVE_UNRANKED_PAIRS,
                )

    return _RankedPairsOnlineDPOTrainer
