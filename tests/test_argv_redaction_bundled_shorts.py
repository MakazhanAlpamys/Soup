"""``mask_secret_args`` must see a secret short option inside a bundle.

Click accepts bundled short options: ``-xtSECRET`` is ``-x`` (a flag) plus
``-t SECRET``. The masker only looked at ``tok[:2]``, so the secret landed in
the audit log verbatim whenever any other short flag was typed in front of it.

No shipped command has a secret-bearing short option today, so this is latent
rather than a live leak — which is the reason to close it now, while the audit
log is the only consumer and nothing depends on the old shape.
"""

from __future__ import annotations

import pytest

from soup_cli.utils.argv_redaction import REDACTED, mask_secret_args

# ``-t`` is the secret-bearing short option throughout; ``-x`` / ``-v`` / ``-q``
# are ordinary flags that happen to be typed in the same bundle.
OPTS = frozenset({"--auth-token", "-t"})


class TestBundledShortOptions:
    def test_secret_bundled_behind_one_flag(self):
        assert mask_secret_args(["-xtSECRET"], OPTS) == (f"-xt{REDACTED}",)

    def test_secret_bundled_behind_several_flags(self):
        assert mask_secret_args(["-xvqtSECRET"], OPTS) == (f"-xvqt{REDACTED}",)

    def test_secret_bundled_with_the_value_in_the_next_token(self):
        assert mask_secret_args(["-xt", "SECRET", "--port", "1"], OPTS) == (
            "-xt", REDACTED, "--port", "1",
        )

    def test_the_secret_never_survives_anywhere_in_the_output(self):
        out = mask_secret_args(["-xtSUPERSECRETVALUE"], OPTS)
        assert not any("SUPERSECRET" in tok for tok in out)

    def test_bundle_of_only_non_secret_flags_is_untouched(self):
        assert mask_secret_args(["-xvq"], OPTS) == ("-xvq",)

    def test_bundle_of_only_non_secret_flags_with_a_value_is_untouched(self):
        assert mask_secret_args(["-xvqVALUE"], OPTS) == ("-xvqVALUE",)

    def test_a_bare_dash_is_untouched(self):
        assert mask_secret_args(["-"], OPTS) == ("-",)

    def test_a_negative_number_argument_is_untouched(self):
        assert mask_secret_args(["--epochs", "-1"], OPTS) == ("--epochs", "-1")

    def test_a_positional_value_is_untouched(self):
        assert mask_secret_args(["train", "config.yaml"], OPTS) == (
            "train", "config.yaml",
        )


class TestTheExistingFormsStillWork:
    """Every shape that worked before keeps working — this is a widening."""

    @pytest.mark.parametrize(
        "argv,expected",
        [
            (["--auth-token", "S"], ("--auth-token", REDACTED)),
            (["--auth-token=S"], (f"--auth-token={REDACTED}",)),
            (["-t", "S"], ("-t", REDACTED)),
            (["-tS"], (f"-t{REDACTED}",)),
            (["-tSECRET", "--port", "1"], (f"-t{REDACTED}", "--port", "1")),
            (["--auth-token"], ("--auth-token",)),
            (["-t"], ("-t",)),
            (["--port", "1"], ("--port", "1")),
        ],
    )
    def test_form(self, argv, expected):
        assert mask_secret_args(argv, OPTS) == expected

    def test_no_secret_options_means_no_masking(self):
        argv = ["-xtSECRET", "--auth-token", "S"]
        assert mask_secret_args(argv, frozenset()) == tuple(argv)
