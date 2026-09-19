"""``plan_modal_run`` must keep ``output_dir`` under the working directory.

``output_dir`` becomes ``_LOCAL_OUTPUT`` in the generated Modal stub — the
local root that ``_download()`` writes the remote run's files into. The stub's
own guard keeps each downloaded *entry* inside that root, but nothing kept the
root itself inside the project: ``config_path`` and ``stub_path`` at the same
call site go through ``enforce_under_cwd_and_no_symlink`` while ``output_dir``
only got a shape check (non-empty, no NUL/newline, length cap).

``cfg.output`` comes from ``soup.yaml``, which is a shareable file whose author
need not be whoever runs it, so an absolute or ``..`` output made a remote
volume's bytes land anywhere the user can write — including ``_download()``
creating the directory on the way.

The Lambda plan reaches the same ``_LOCAL_OUTPUT`` slot; its
``_validate_remote_output`` already rejects absolute and ``..`` paths by shape,
so the guard there adds only the symlink / junction rejection.
"""

from __future__ import annotations

import os

import pytest

_SOUP_YAML = """base: hf-internal-testing/tiny-random-gpt2
task: sft
data:
  train: data.jsonl
training:
  epochs: 1
output: ./output
"""


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
    return tmp_path


def _plan(output_dir: str):
    from soup_cli.cloud.modal import plan_modal_run

    return plan_modal_run(
        "soup.yaml", gpu="a100", output_dir=output_dir, soup_version="0.75.1"
    )


class TestModalOutputDirContainment:
    def test_absolute_output_dir_is_refused(self, project, tmp_path):
        outside = str(tmp_path.parent / "elsewhere")
        assert os.path.isabs(outside)

        with pytest.raises(ValueError, match="output_dir"):
            _plan(outside)

    def test_absolute_refusal_says_it_must_stay_under_the_working_dir(
        self, project, tmp_path
    ):
        with pytest.raises(ValueError) as excinfo:
            _plan(str(tmp_path.parent / "elsewhere"))
        message = str(excinfo.value)
        assert "output_dir" in message
        assert "cwd" in message or "working directory" in message

    def test_dotdot_output_dir_is_refused(self, project):
        with pytest.raises(ValueError, match="output_dir"):
            _plan("../escape")

    def test_nested_dotdot_output_dir_is_refused(self, project):
        with pytest.raises(ValueError, match="output_dir"):
            _plan("out/../../escape")

    def test_symlinked_output_dir_is_refused(self, project, tmp_path):
        target = tmp_path.parent / "link-target"
        target.mkdir(exist_ok=True)
        try:
            os.symlink(str(target), str(project / "out"), target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError) as exc:
            pytest.skip(f"symlinks unavailable on this machine: {exc}")

        with pytest.raises(ValueError, match="output_dir"):
            _plan("out")

    def test_the_refusal_happens_before_any_stub_is_written(self, project):
        with pytest.raises(ValueError):
            _plan("../escape")
        assert not (project / "soup_modal_app.py").exists()


class TestTheDefaultStillWorks:
    def test_default_relative_output_plans(self, project):
        plan = _plan("./output")

        assert plan.cloud == "modal"
        assert plan.output_dir == "./output"
        assert "modal.App" in plan.stub_text

    def test_nested_relative_output_plans(self, project):
        assert _plan("runs/exp1").output_dir == "runs/exp1"

    def test_the_generated_stub_is_unchanged_for_the_default(self, project):
        from soup_cli.cloud.modal import render_modal_stub

        plan = _plan("./output")
        expected = render_modal_stub(
            _SOUP_YAML, gpu="a100", output_dir="./output",
            soup_version="0.75.1",
            run_name=plan.stub_text.split("_RUN_NAME = ")[1].split("\n")[0].strip("'\""),
        )
        assert plan.stub_text == expected
        assert "_LOCAL_OUTPUT = './output'" in plan.stub_text

    def test_an_existing_real_output_directory_is_fine(self, project):
        (project / "output").mkdir()

        assert _plan("./output").output_dir == "./output"


class TestLambdaOutputDirContainment:
    """Lambda's shape check already rejects absolute / ``..``; the added guard
    is the symlink case. Both are asserted so a loosened shape check is seen."""

    def _plan_lambda(self, output_dir: str):
        from soup_cli.cloud.lambda_labs import plan_lambda_run

        return plan_lambda_run(
            "soup.yaml", gpu="a100", output_dir=output_dir,
            soup_version="0.75.1",
        )

    def test_default_relative_output_plans(self, project):
        assert self._plan_lambda("out").output_dir == "out"

    def test_absolute_output_dir_is_refused(self, project, tmp_path):
        with pytest.raises(ValueError, match="output_dir"):
            self._plan_lambda(str(tmp_path.parent / "elsewhere"))

    def test_dotdot_output_dir_is_refused(self, project):
        with pytest.raises(ValueError, match="output_dir"):
            self._plan_lambda("../escape")

    def test_symlinked_output_dir_is_refused(self, project, tmp_path):
        target = tmp_path.parent / "lambda-link-target"
        target.mkdir(exist_ok=True)
        try:
            os.symlink(str(target), str(project / "out"), target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError) as exc:
            pytest.skip(f"symlinks unavailable on this machine: {exc}")

        with pytest.raises(ValueError, match="output_dir"):
            self._plan_lambda("out")
