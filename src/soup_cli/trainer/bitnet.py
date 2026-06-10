"""v0.71.20 #134 — Live BitNet 1.58-bit fine-tuning trainer (BETA, hw-gated).

``BitNetTrainerWrapper`` lifts the v0.52.0 ``build_bitnet_trainer`` schema
stub. BitNet 1.58 fine-tuning trains an SFT-style next-token cross-entropy
objective on a model whose ``BitLinear`` layers carry ternary weights. The
faithful training path needs the upstream ``onebitllms`` package, which wraps
the BitLinear gradient + ternary quantization (CUDA / Linux only).

This wrapper reuses :class:`~soup_cli.trainer.sft.SFTTrainerWrapper` for the
data/CE/LoRA machinery and gates :meth:`setup` on ``onebitllms`` availability:
when it is absent the wrapper raises a friendly ``RuntimeError`` naming the
package rather than a bare ``NotImplementedError``. The real end-to-end
training run is hardware/dependency-gated and stays an open infra-blocked item
(no ``onebitllms`` on the maintainer's Windows box).
"""

from __future__ import annotations

from soup_cli.trainer.sft import SFTTrainerWrapper, console


class BitNetTrainerWrapper(SFTTrainerWrapper):
    """SFT-based BitNet 1.58-bit trainer (v0.71.20 #134).

    Inherits the SFT setup/train machinery and gates :meth:`setup` on
    ``onebitllms``. The actual BitLinear-aware training is delegated to the
    upstream package; absent it, a friendly error fires.
    """

    def setup(self, dataset: dict) -> None:
        self._require_onebitllms()
        # onebitllms IS importable (never validated on the maintainer's box —
        # no Windows wheel). The faithful path would patch the model's
        # BitLinear layers via onebitllms before the standard SFT CE loop;
        # surface it as an explicit not-yet-validated error so we never ship an
        # unrun training path silently.
        raise RuntimeError(
            "BitNet 1.58-bit fine-tuning is hardware/dependency-gated. The "
            "'onebitllms' package is importable but the BitLinear-aware "
            "training path has not been validated on this platform. Run on a "
            "CUDA/Linux host with onebitllms + a BitNet base model."
        )

    def _require_onebitllms(self) -> None:
        """Raise a friendly error when the ``onebitllms`` package is absent."""
        import importlib.util

        if importlib.util.find_spec("onebitllms") is None:
            console.print(
                "[yellow]BitNet 1.58-bit fine-tuning requires the "
                "'onebitllms' package.[/]"
            )
            raise RuntimeError(
                "BitNet 1.58-bit fine-tuning (quantization='bitnet_1.58') "
                "requires the 'onebitllms' package, which ships the "
                "BitLinear-aware training kernels. Install it with "
                "`pip install onebitllms` (CUDA / Linux only)."
            )
