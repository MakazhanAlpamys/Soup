"""#144 — two defects found by the first CUDA-box run of `soup export --format gguf`.

All six quantisation tiers produce artifacts that load in `llama-cli` and emit
correct text, which is the headline result. These are the two things that went
wrong around that.

**G1 is data loss.** The f16 intermediate was named
``{model_name}.f16.gguf`` — which is EXACTLY the default output name of a
``--quant f16`` export — and deleted when the quantisation finished. So exporting
q4_0 destroyed a previously exported f16 GGUF, even with an unrelated
``--output``. Reproduced on the box.

**G2** — ``_install_convert_deps()`` ran only in the auto-clone branch, so anyone
passing ``--llama-cpp`` or already holding ``~/.soup/llama.cpp`` hit
``ModuleNotFoundError: No module named 'sentencepiece'`` on every conversion.
"""



def _run_export(model_dir, output):
    """Drive the real CLI.

    Calling the Typer command as a plain function leaves every unset option as an
    `OptionInfo` sentinel rather than its default, which fails inside the command
    for reasons that have nothing to do with what is being tested.
    """
    from typer.testing import CliRunner

    from soup_cli.cli import app

    result = CliRunner().invoke(
        app,
        ["export", "--model", str(model_dir), "--format", "gguf",
         "--quant", "q4_0", "--output", str(output)],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return result



class TestTheF16IntermediateCannotEatAUsersFile:
    def _fake_llama_cpp(self, tmp_path):
        llama = tmp_path / "llama.cpp"
        llama.mkdir()
        (llama / "convert_hf_to_gguf.py").write_text("# stub\n", encoding="utf-8")
        return llama

    def _model_dir(self, tmp_path):
        model = tmp_path / "mymodel"
        model.mkdir()
        (model / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
        return model

    def test_a_quantised_export_does_not_delete_an_existing_f16_gguf(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.commands import export as export_mod

        model = self._model_dir(tmp_path)
        llama = self._fake_llama_cpp(tmp_path)

        # the artifact a previous `--quant f16` export produced, at its DEFAULT name
        victim = tmp_path / "mymodel.f16.gguf"
        victim.write_bytes(b"a real gguf the user still wants")

        monkeypatch.setattr(export_mod, "_find_llama_cpp", lambda p=None: llama)
        monkeypatch.setattr(
            export_mod, "_run_convert",
            lambda script, src, dst, outtype: dst.write_bytes(b"intermediate"),
        )
        monkeypatch.setattr(
            export_mod, "_run_quantize",
            lambda ldir, src, dst, quant: dst.write_bytes(b"quantised"),
        )

        _run_export(model, tmp_path / "out.q4_0.gguf")

        assert victim.exists(), (
            "the quantised export deleted the user's previously exported f16 GGUF — "
            "the intermediate is colliding with a real output name (#144 G1)"
        )
        assert victim.read_bytes() == b"a real gguf the user still wants"

    def test_the_intermediate_is_still_cleaned_up(self, tmp_path, monkeypatch):
        """CONTROL. Fixing the collision must not leave multi-gigabyte f16 files
        behind — that would trade data loss for a full disk."""
        from soup_cli.commands import export as export_mod

        model = self._model_dir(tmp_path)
        llama = self._fake_llama_cpp(tmp_path)
        monkeypatch.setattr(export_mod, "_find_llama_cpp", lambda p=None: llama)
        monkeypatch.setattr(
            export_mod, "_run_convert",
            lambda script, src, dst, outtype: dst.write_bytes(b"intermediate"),
        )
        monkeypatch.setattr(
            export_mod, "_run_quantize",
            lambda ldir, src, dst, quant: dst.write_bytes(b"quantised"),
        )

        out = tmp_path / "out.q4_0.gguf"
        _run_export(model, out)

        assert out.exists()
        leftovers = [
            p for p in tmp_path.rglob("*.f16.gguf") if p.name != "mymodel.f16.gguf"
        ]
        assert leftovers == [], f"intermediate f16 left behind: {leftovers}"


class TestConvertDepsAreInstalledOnEveryPath:
    def test_a_user_supplied_llama_cpp_still_gets_the_convert_deps(
        self, tmp_path, monkeypatch
    ):
        """`--llama-cpp /path` and an already-cloned ~/.soup/llama.cpp both skipped
        the dependency install, so every conversion died on `sentencepiece`.

        Asserted through the EXPORT FLOW rather than through `_find_llama_cpp`,
        because that is where the install now lives: a lookup function that
        silently shells out to pip is a surprise, and an existing test that mocks
        `subprocess.run` to catch clone attempts saw the pip call instead.
        """
        from soup_cli.commands import export as export_mod

        model = tmp_path / "mymodel"
        model.mkdir()
        (model / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
        llama = tmp_path / "llama.cpp"
        llama.mkdir()
        (llama / "convert_hf_to_gguf.py").write_text("# stub", encoding="utf-8")

        calls = []
        monkeypatch.setattr(export_mod, "_find_llama_cpp", lambda p=None: llama)
        monkeypatch.setattr(export_mod, "_install_convert_deps", lambda: calls.append(1))
        monkeypatch.setattr(
            export_mod, "_run_convert",
            lambda script, src, dst, outtype: dst.write_bytes(b"i"),
        )
        monkeypatch.setattr(
            export_mod, "_run_quantize",
            lambda ldir, src, dst, quant: dst.write_bytes(b"q"),
        )

        _run_export(model, tmp_path / "out.q4_0.gguf")
        assert calls, (
            "the export never installed the convert dependencies, so a conversion "
            "on a machine without sentencepiece dies (#144 G2)"
        )

    def test_looking_for_llama_cpp_does_not_shell_out(self, tmp_path, monkeypatch):
        """CONTROL, and a regression guard for how this was first fixed wrongly.

        `_find_llama_cpp` must stay a lookup. The first version of the fix called
        the installer from inside it, which on a machine WITHOUT the packages
        turned an existing clone-detection test into a false failure — and that
        machine was CI, not this one.
        """
        from soup_cli.commands import export as export_mod

        llama = tmp_path / "llama.cpp"
        llama.mkdir()
        (llama / "convert_hf_to_gguf.py").write_text("# stub", encoding="utf-8")

        def _explode(*args, **kwargs):
            raise AssertionError("_find_llama_cpp must not run a subprocess")

        monkeypatch.setattr(export_mod.subprocess, "run", _explode)
        assert export_mod._find_llama_cpp(str(llama)) == llama

    def test_the_install_is_skipped_when_the_deps_are_already_importable(
        self, tmp_path, monkeypatch
    ):
        """CONTROL. It must not shell out to pip on every export when the packages
        are already present — that is a network call in the hot path."""
        from soup_cli.commands import export as export_mod

        calls = []
        monkeypatch.setattr(export_mod.subprocess, "run", lambda *a, **k: calls.append(a))
        monkeypatch.setattr(export_mod, "_convert_deps_present", lambda: True)
        export_mod._install_convert_deps()
        assert calls == [], "pip was invoked although the deps were already importable"

    def test_the_install_runs_when_they_are_not(self, tmp_path, monkeypatch):
        """The other half. Without this, pinning "does not call pip" would be
        satisfied by an installer that never installs anything."""
        from soup_cli.commands import export as export_mod

        calls = []
        monkeypatch.setattr(export_mod.subprocess, "run", lambda *a, **k: calls.append(a))
        monkeypatch.setattr(export_mod, "_convert_deps_present", lambda: False)
        export_mod._install_convert_deps()
        assert calls, "the deps were missing and pip was never invoked"
