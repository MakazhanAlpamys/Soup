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
        with pytest.raises(ValueError, match="got bool"):
            validate_cloud(True)
        with pytest.raises(ValueError, match="got int"):
            validate_cloud(123)
        with pytest.raises(ValueError, match="non-empty string"):
            validate_cloud("")
        with pytest.raises(ValueError, match="null bytes"):
            validate_cloud("run\x00pod")
        with pytest.raises(ValueError, match="exceeds"):
            validate_cloud("a" * 100)

    def test_runpod_validate_gpu(self):
        from soup_cli.cloud.runpod import validate_gpu

        assert validate_gpu("rtx-4090") == "rtx-4090"
        with pytest.raises(ValueError):
            validate_gpu("tpu")
        with pytest.raises(ValueError, match="got bool"):
            validate_gpu(True)
        with pytest.raises(ValueError, match="got int"):
            validate_gpu(123)
        with pytest.raises(ValueError, match="non-empty string"):
            validate_gpu("")
        with pytest.raises(ValueError, match="null bytes"):
            validate_gpu("rtx\x004090")
        with pytest.raises(ValueError, match="exceeds"):
            validate_gpu("a" * 100)


class TestRenderRunpodStub:
    def test_render_happy_path(self):
        from soup_cli.cloud.runpod import render_runpod_stub

        stub = render_runpod_stub(
            _SOUP_YAML, gpu="a100", output_dir="./out", soup_version="0.71.22"
        )
        assert "runpod.create_pod" in stub
        assert "NVIDIA A100 80GB PCIe" in stub
        assert "soup-cli[train]==0.71.22" in stub
        assert "RUNPOD_NETWORK_VOLUME_ID" in stub
        assert 'volume_mount_path="/workspace"' in stub
        assert "cd /workspace" in stub

    def test_bad_output_dir_rejected(self):
        from soup_cli.cloud.runpod import render_runpod_stub

        with pytest.raises(ValueError):
            render_runpod_stub(
                _SOUP_YAML, gpu="a100", output_dir="out\nINJECT", soup_version="0.71.22"
            )

    @pytest.mark.parametrize("output_dir", ["/tmp/out", "../out", "out with spaces"])
    def test_non_persistent_or_unsafe_output_rejected(self, output_dir):
        from soup_cli.cloud.runpod import render_runpod_stub

        with pytest.raises(ValueError, match="output_dir"):
            render_runpod_stub(
                _SOUP_YAML, gpu="a100", output_dir=output_dir, soup_version="0.71.22"
            )

    def test_render_config_yaml_validation(self):
        from soup_cli.cloud.runpod import render_runpod_stub

        with pytest.raises(TypeError, match="must be a string"):
            render_runpod_stub(123, gpu="a100", output_dir="./out", soup_version="1.0.0")
        with pytest.raises(ValueError, match="exceeds"):
            render_runpod_stub(
                "a" * 2_000_000, gpu="a100", output_dir="./out", soup_version="1.0.0"
            )

    def test_render_soup_version_validation(self):
        from soup_cli.cloud.runpod import render_runpod_stub

        with pytest.raises(ValueError, match="NUL-free string"):
            render_runpod_stub(_SOUP_YAML, gpu="a100", output_dir="./out", soup_version="1.0\x00")
        with pytest.raises(ValueError, match="must match"):
            render_runpod_stub(_SOUP_YAML, gpu="a100", output_dir="./out", soup_version="invalid!")
        with pytest.raises(ValueError, match="must match"):
            render_runpod_stub(_SOUP_YAML, gpu="a100", output_dir="./out", soup_version="a" * 100)


class TestPlanRunpodRun:
    def test_plan(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
        from soup_cli.cloud._common import CloudPlan
        from soup_cli.cloud.runpod import plan_runpod_run

        plan = plan_runpod_run("soup.yaml", gpu="a100", output_dir="./out", soup_version="0.71.22")
        assert isinstance(plan, CloudPlan)
        assert plan.cloud == "runpod"
        assert plan.gpu == "a100"
        assert plan.run_command == "python soup_runpod_app.py"

    def test_plan_config_too_large(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text("a" * 2_000_000, encoding="utf-8")
        from soup_cli.cloud.runpod import plan_runpod_run

        with pytest.raises(ValueError, match="exceeds"):
            plan_runpod_run("soup.yaml", gpu="a100", output_dir="./out", soup_version="0.71.22")


class TestSubmitRunpodRun:
    def test_override_seam(self, monkeypatch):
        import soup_cli.cloud.runpod as m
        from soup_cli.cloud._common import CloudPlan

        plan = CloudPlan(
            cloud="runpod",
            gpu="a100",
            output_dir="./out",
            stub_path="x.py",
            stub_text="",
            run_command="python x.py",
        )
        monkeypatch.setattr(m, "_RUNPOD_SUBMIT_OVERRIDE", lambda p: 7)
        assert m.submit_runpod_run(plan) == 7

    def test_no_token_raises(self):
        import soup_cli.cloud.runpod as m
        from soup_cli.cloud._common import CloudPlan

        plan = CloudPlan(
            cloud="runpod",
            gpu="a100",
            output_dir="./out",
            stub_path="x.py",
            stub_text="",
            run_command="python x.py",
        )
        with pytest.raises(RuntimeError, match="not authenticated"):
            m.submit_runpod_run(plan, env={})

    def test_runpod_sdk_missing_raises(self, monkeypatch):
        import sys

        import soup_cli.cloud.runpod as m
        from soup_cli.cloud._common import CloudPlan

        plan = CloudPlan(
            cloud="runpod",
            gpu="a100",
            output_dir="./out",
            stub_path="x.py",
            stub_text="",
            run_command="python x.py",
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
        with pytest.raises(ValueError, match="got bool"):
            validate_cloud(True)
        with pytest.raises(ValueError, match="got int"):
            validate_cloud(123)
        with pytest.raises(ValueError, match="non-empty string"):
            validate_cloud("")
        with pytest.raises(ValueError, match="null bytes"):
            validate_cloud("lamb\x00da")
        with pytest.raises(ValueError, match="exceeds"):
            validate_cloud("a" * 100)

    def test_lambda_validate_gpu(self):
        from soup_cli.cloud.lambda_labs import validate_gpu

        assert validate_gpu("a100") == "a100"
        with pytest.raises(ValueError):
            validate_gpu("tpu")
        with pytest.raises(ValueError, match="got bool"):
            validate_gpu(True)
        with pytest.raises(ValueError, match="got int"):
            validate_gpu(123)
        with pytest.raises(ValueError, match="non-empty string"):
            validate_gpu("")
        with pytest.raises(ValueError, match="null bytes"):
            validate_gpu("a10\x000")
        with pytest.raises(ValueError, match="exceeds"):
            validate_gpu("a" * 100)


class TestRenderLambdaLabsStub:
    def test_render_happy_path(self):
        from soup_cli.cloud.lambda_labs import render_lambda_stub

        stub = render_lambda_stub(
            _SOUP_YAML, gpu="a100", output_dir="./out", soup_version="0.71.22"
        )
        assert "urllib.request" in stub
        assert "gpu_1x_a100_sxm4" in stub
        compile(stub, "soup_lambda_app.py", "exec")
        namespace = {"__name__": "test"}
        exec(stub, namespace)
        payload = namespace["_launch_payload"]("us-tx-1", "registered-key")
        assert payload["user_data"] == namespace["_USER_DATA"]
        assert "soup-cli[train]==0.71.22" in payload["user_data"]
        assert "LAMBDA_API_KEY" not in payload["user_data"]
        assert payload["ssh_key_names"] == ["registered-key"]
        assert "quantity" not in payload

    @pytest.mark.parametrize("output_dir", ["/tmp/out", "../out", "out with spaces"])
    def test_non_retrievable_or_unsafe_output_rejected(self, output_dir):
        from soup_cli.cloud.lambda_labs import render_lambda_stub

        with pytest.raises(ValueError, match="output_dir"):
            render_lambda_stub(
                _SOUP_YAML, gpu="a100", output_dir=output_dir, soup_version="0.71.22"
            )

    def test_generated_controller_always_terminates(self, tmp_path, monkeypatch):
        from soup_cli.cloud.lambda_labs import render_lambda_stub

        key = tmp_path / "lambda-key"
        key.write_text("test-only", encoding="utf-8")
        stub = render_lambda_stub(
            _SOUP_YAML, gpu="a100", output_dir="./out", soup_version="0.71.22"
        )
        namespace = {"__name__": "test"}
        exec(stub, namespace)
        requests = []

        def fake_request(api_key, path, *, method="GET", payload=None):
            requests.append((api_key, path, method, payload))
            if path.endswith("/launch"):
                return {"data": {"instance_ids": ["instance-1"]}}
            return {"data": {}}

        monkeypatch.setenv("LAMBDA_API_KEY", "secret")
        monkeypatch.setenv("LAMBDA_SSH_KEY_NAME", "registered-key")
        monkeypatch.setenv("LAMBDA_SSH_PRIVATE_KEY", str(key))
        namespace["_request_json"] = fake_request
        namespace["_wait_for_ip"] = lambda *_: "192.0.2.1"
        termination_checks = []
        namespace["_wait_for_termination"] = lambda *args: termination_checks.append(args)
        namespace["_train_and_copy"] = lambda *_: (_ for _ in ()).throw(
            RuntimeError("training failed")
        )

        assert namespace["main"]() == 1
        assert termination_checks == [("secret", "instance-1")]
        assert requests[-1][1:] == (
            "/instance-operations/terminate",
            "POST",
            {"instance_ids": ["instance-1"]},
        )

    def test_render_config_yaml_validation(self):
        from soup_cli.cloud.lambda_labs import render_lambda_stub

        with pytest.raises(TypeError, match="must be a string"):
            render_lambda_stub(123, gpu="a100", output_dir="./out", soup_version="1.0.0")
        with pytest.raises(ValueError, match="exceeds"):
            render_lambda_stub(
                "a" * 2_000_000, gpu="a100", output_dir="./out", soup_version="1.0.0"
            )

    def test_render_soup_version_validation(self):
        from soup_cli.cloud.lambda_labs import render_lambda_stub

        with pytest.raises(ValueError, match="NUL-free string"):
            render_lambda_stub(_SOUP_YAML, gpu="a100", output_dir="./out", soup_version="1.0\x00")
        with pytest.raises(ValueError, match="must match"):
            render_lambda_stub(_SOUP_YAML, gpu="a100", output_dir="./out", soup_version="invalid!")
        with pytest.raises(ValueError, match="must match"):
            render_lambda_stub(_SOUP_YAML, gpu="a100", output_dir="./out", soup_version="a" * 100)


class TestPlanLambdaLabsRun:
    def test_plan(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
        from soup_cli.cloud._common import CloudPlan
        from soup_cli.cloud.lambda_labs import plan_lambda_run

        plan = plan_lambda_run("soup.yaml", gpu="a100", output_dir="./out", soup_version="0.71.22")
        assert isinstance(plan, CloudPlan)
        assert plan.cloud == "lambda"
        assert plan.run_command == "python soup_lambda_app.py"

    def test_plan_config_too_large(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text("a" * 2_000_000, encoding="utf-8")
        from soup_cli.cloud.lambda_labs import plan_lambda_run

        with pytest.raises(ValueError, match="exceeds"):
            plan_lambda_run("soup.yaml", gpu="a100", output_dir="./out", soup_version="0.71.22")


class TestSubmitLambdaLabsRun:
    def test_override_seam(self, monkeypatch):
        import soup_cli.cloud.lambda_labs as m
        from soup_cli.cloud._common import CloudPlan

        plan = CloudPlan(
            cloud="lambda",
            gpu="a100",
            output_dir="./out",
            stub_path="x.py",
            stub_text="",
            run_command="python x.py",
        )
        monkeypatch.setattr(m, "_LAMBDA_SUBMIT_OVERRIDE", lambda p: 7)
        assert m.submit_lambda_run(plan) == 7

    def test_no_token_raises(self):
        import soup_cli.cloud.lambda_labs as m
        from soup_cli.cloud._common import CloudPlan

        plan = CloudPlan(
            cloud="lambda",
            gpu="a100",
            output_dir="./out",
            stub_path="x.py",
            stub_text="",
            run_command="python x.py",
        )
        with pytest.raises(RuntimeError, match="not authenticated/configured"):
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

    def test_cloud_name_is_case_insensitive(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_SOUP_YAML, encoding="utf-8")
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(
            app, ["train", "--config", "soup.yaml", "--cloud", "RUNPOD", "--gpu", "a100"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "soup_runpod_app.py").exists()


class TestCloudNoTopLevelSDK:
    def test_no_top_level_runpod_import(self):
        src = (REPO_ROOT / "src/soup_cli/cloud/runpod.py").read_text(encoding="utf-8")
        assert "\nimport runpod\n" not in src

    def test_runpod_extra_in_pyproject(self):
        pp = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "runpod = [" in pp
