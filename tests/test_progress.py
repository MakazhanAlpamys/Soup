"""Tests for Rich download progress bar integration."""


def test_enable_hf_transfer_progress_no_crash():
    """Calling _enable_hf_transfer_progress should not raise."""
    from soup_cli.trainer.sft import _enable_hf_transfer_progress

    # Should not raise even if huggingface_hub internals change
    _enable_hf_transfer_progress()


def test_enable_hf_transfer_progress_idempotent():
    """Calling twice should not raise."""
    from soup_cli.trainer.sft import _enable_hf_transfer_progress

    _enable_hf_transfer_progress()
    _enable_hf_transfer_progress()
