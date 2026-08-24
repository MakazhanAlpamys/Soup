from pathlib import Path

import pytest

_SOUP_YAML = (
    "base: hf-internal-testing/tiny-random-gpt2\n"
    "task: sft\n"
    "data:\n  train: data.jsonl\n  format: chatml\n"
    "output: ./out\n"
)

REPO_ROOT = Path(__file__).parent.parent.resolve()

def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", s)


class TestValidateRunpod:
    def test_runpod_validate_cloud(self):
        from soup_cli.cloud.runpod import validate_cloud

        assert validate_cloud("runpod") == "runpod"
        assert validate_cloud("RUNPOD") == "runpod"
        with pytest.raises(ValueError):
            validate_cloud("modal")

    def test_runpod_validate_gpu(self):
        from soup_cli.cloud.runpod import validate_gpu

        assert validate_gpu("rtx-4090") == "rtx-4090"
        with pytest.raises(ValueError):
            validate_gpu("tpu")


class TestRenderRunpodStub:
    def test_render_happy_path(self):
        from soup_cli.cloud.runpod import render_runpod_stub

        stub = render_runpod_stub(
            _SOUP_YAML, gpu="a100", output_dir="./out", soup_version="0.71.22"
        )
        assert "runpod.create_pod" in stub
        assert "NVIDIA A100 80GB PCIe" in stub
        assert "soup-cli[train]==0.71.22" in stub

    def test_bad_output_dir_rejected(self):
        from soup_cli.cloud.runpod import render_runpod_stub

        with pytest.raises(ValueError):
            render_runpod_stub(
                _SOUP_YAML, gpu="a100", output_dir="out\nINJECT", soup_version="0.71.22"
            )


class TestPlanRunpodRun:
    def test_plan(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
        from soup_cli.cloud.modal import CloudPlan
        from soup_cli.cloud.runpod import plan_runpod_run

        plan = plan_runpod_run(
            "soup.yaml", gpu="a100", output_dir="./out", soup_version="0.71.22"
        )
        assert isinstance(plan, CloudPlan)
        assert plan.cloud == "runpod"
        assert plan.gpu == "a100"
        assert plan.run_command == "python soup_runpod_app.py"


class TestSubmitRunpodRun:
    def test_override_seam(self, monkeypatch):
        import soup_cli.cloud.runpod as m
        from soup_cli.cloud.modal import CloudPlan

        plan = CloudPlan(
            cloud="runpod", gpu="a100", output_dir="./out",
            stub_path="x.py", stub_text="", run_command="python x.py",
        )
        monkeypatch.setattr(m, "_RUNPOD_SUBMIT_OVERRIDE", lambda p: 7)
        assert m.submit_runpod_run(plan) == 7

    def test_no_token_raises(self):
        import soup_cli.cloud.runpod as m
        from soup_cli.cloud.modal import CloudPlan

        plan = CloudPlan(
            cloud="runpod", gpu="a100", output_dir="./out",
            stub_path="x.py", stub_text="", run_command="python x.py",
        )
        with pytest.raises(RuntimeError, match="not authenticated"):
            m.submit_runpod_run(plan, env={})

    def test_runpod_sdk_missing_raises(self, monkeypatch):
        import sys

        import soup_cli.cloud.runpod as m
        from soup_cli.cloud.modal import CloudPlan

        plan = CloudPlan(
            cloud="runpod", gpu="a100", output_dir="./out",
            stub_path="x.py", stub_text="", run_command="python x.py",
        )
        monkeypatch.setitem(sys.modules, "runpod", None)
        with pytest.raises(RuntimeError, match="not installed"):
            m.submit_runpod_run(plan, env={"RUNPOD_API_KEY": "a"})


class TestValidateLambdaLabs:
    def test_lambda_validate_cloud(self):
        from soup_cli.cloud.lambda_labs import validate_cloud

        assert validate_cloud("lambda") == "lambda"
        with pytest.raises(ValueError):
            validate_cloud("runpod")

    def test_lambda_validate_gpu(self):
        from soup_cli.cloud.lambda_labs import validate_gpu

        assert validate_gpu("a100") == "a100"
        with pytest.raises(ValueError):
            validate_gpu("tpu")


class TestRenderLambdaLabsStub:
    def test_render_happy_path(self):
        from soup_cli.cloud.lambda_labs import render_lambda_stub

        stub = render_lambda_stub(
            _SOUP_YAML, gpu="a100", output_dir="./out", soup_version="0.71.22"
        )
        assert "urllib.request" in stub
        assert "gpu_1x_a100_sxm4" in stub

        import base64
        import re
        match = re.search(r'user_data_b64 = "([^"]+)"', stub)
        assert match is not None
        user_data = base64.b64decode(match.group(1)).decode("utf-8")
        assert "soup-cli[train]==0.71.22" in user_data


class TestPlanLambdaLabsRun:
    def test_plan(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
        from soup_cli.cloud.lambda_labs import plan_lambda_run
        from soup_cli.cloud.modal import CloudPlan

        plan = plan_lambda_run(
            "soup.yaml", gpu="a100", output_dir="./out", soup_version="0.71.22"
        )
        assert isinstance(plan, CloudPlan)
        assert plan.cloud == "lambda"
        assert plan.run_command == "python soup_lambda_app.py"


class TestSubmitLambdaLabsRun:
    def test_override_seam(self, monkeypatch):
        import soup_cli.cloud.lambda_labs as m
        from soup_cli.cloud.modal import CloudPlan

        plan = CloudPlan(
            cloud="lambda", gpu="a100", output_dir="./out",
            stub_path="x.py", stub_text="", run_command="python x.py",
        )
        monkeypatch.setattr(m, "_LAMBDA_SUBMIT_OVERRIDE", lambda p: 7)
        assert m.submit_lambda_run(plan) == 7

    def test_no_token_raises(self):
        import soup_cli.cloud.lambda_labs as m
        from soup_cli.cloud.modal import CloudPlan

        plan = CloudPlan(
            cloud="lambda", gpu="a100", output_dir="./out",
            stub_path="x.py", stub_text="", run_command="python x.py",
        )
        with pytest.raises(RuntimeError, match="not authenticated"):
            m.submit_lambda_run(plan, env={})


class TestTrainCloudCliNew:
    def test_cloud_runpod_plan_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(
            app, ["train", "--config", "soup.yaml", "--cloud", "runpod", "--gpu", "rtx-4090"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "soup_runpod_app.py").exists()
        txt = _strip_ansi(result.output)
        assert "python soup_runpod_app.py" in txt
        assert "plan-only" in txt.lower()

    def test_cloud_lambda_plan_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(
            app, ["train", "--config", "soup.yaml", "--cloud", "lambda", "--gpu", "a10"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "soup_lambda_app.py").exists()
        txt = _strip_ansi(result.output)
        assert "python soup_lambda_app.py" in txt


class TestCloudNoTopLevelSDK:
    def test_no_top_level_runpod_import(self):
        src = (REPO_ROOT / "src/soup_cli/cloud/runpod.py").read_text(encoding="utf-8")
        assert "\nimport runpod\n" not in src

    def test_runpod_extra_in_pyproject(self):
        pp = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "runpod = [" in pp
