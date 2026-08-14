"""`device_map="auto"` is wrong under any distributed launch, and it was everywhere.

Found on 8x H100 (benchmarks/gate-h100-validation.md, FINDING 4). The command
`soup train --gpus 8` prints for the user --

    accelerate launch --num_processes 8 soup train -c config.yaml

-- failed on every rank before training started:

    ValueError: You can't train a model that has been loaded with
    `device_map='auto'` in any distributed mode.

`device_map="auto"` shards one model across every visible GPU; under torchrun /
accelerate / deepspeed every rank would try to do that. Six trainers carried the
same unguarded line. So the multi-GPU path the tool advertises had never worked
on the SFT path, and a one-GPU dev box cannot see it.

These tests run everywhere: the distributed environment is just env vars.
"""

import pathlib
import re

import pytest

from soup_cli.utils.gpu import resolve_device_map

_TRAINER_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "trainer"
_TRAINER_SOURCES = sorted(_TRAINER_DIR.glob("*.py"))

#: The exact idiom this fix replaces: ``dev_map = "cpu" if <cond> else "auto"``.
#: Written against the assignment rather than a variable name, so a copy that
#: renames ``dev_map`` is still caught.
_OLD_IDIOM = re.compile(r"""=\s*["']cpu["']\s+if\s+.+?\s+else\s+["']auto["']""")
#: The other spelling PPO used: ``device_map="auto" if ... else None``.
_INLINE_AUTO = re.compile(r"""device_map\s*=\s*["']auto["']""")


def _code_without_comments(text: str) -> str:
    """Strip trailing comments so the stale ``# ... from "auto"`` notes left
    above the fixed lines do not read as violations."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Real torchrun vars leak in when the suite itself runs under a launcher."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


class TestSingleProcess:
    def test_cpu_stays_cpu(self):
        """'auto' on CPU produces meta tensors -- the reason the original line
        special-cased CPU at all. That behaviour must survive the fix."""
        assert resolve_device_map("cpu") == "cpu"

    def test_gpu_without_a_launcher_is_auto(self):
        assert resolve_device_map("cuda") == "auto"

    def test_world_size_one_is_auto_even_with_local_rank_set(self, monkeypatch):
        """`accelerate launch --num_processes 1` sets LOCAL_RANK=0 and
        WORLD_SIZE=1. That is a single process, and pinning it to {"" : 0} would
        needlessly disable auto-placement."""
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        assert resolve_device_map("cuda") == "auto"


class TestDistributed:
    @pytest.mark.parametrize("rank", [0, 1, 7])
    def test_each_rank_pins_its_own_device(self, monkeypatch, rank):
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", "8")
        assert resolve_device_map("cuda") == {"": rank}

    def test_cpu_wins_over_a_distributed_environment(self, monkeypatch):
        """--device cpu under a launcher must not be handed a CUDA index."""
        monkeypatch.setenv("LOCAL_RANK", "3")
        monkeypatch.setenv("WORLD_SIZE", "8")
        assert resolve_device_map("cpu") == "cpu"

    def test_never_returns_auto_when_distributed(self, monkeypatch):
        """The whole defect in one assertion: transformers raises on 'auto' in
        any distributed mode, so this is the value that must never come back."""
        monkeypatch.setenv("LOCAL_RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "4")
        assert resolve_device_map("cuda") != "auto"


class TestMalformedEnvironment:
    @pytest.mark.parametrize("world", ["", "junk", "1.5"])
    def test_unparsable_world_size_falls_back_to_auto(self, monkeypatch, world):
        """A process with no usable distributed environment IS single-process."""
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", world)
        assert resolve_device_map("cuda") == "auto"

    def test_unparsable_local_rank_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "junk")
        monkeypatch.setenv("WORLD_SIZE", "8")
        assert resolve_device_map("cuda") == "auto"


class TestNoTrainerStillHardcodesAuto:
    """Guards the COVERAGE, and derives it from the sources rather than a list.

    The first version of this class parametrized over six hand-written trainer
    names -- exactly the six the original fix had touched -- and passed while
    NINE more sites still carried the old line: bco, distill, embedding, ipo,
    online_dpo, orpo, reward_model, simpo and PPO's own reward-model loader. So
    `accelerate launch` still died on those tasks, and the guard that existed to
    notice could not, because a hand-written list cannot report what it does not
    name. The scan below covers every module in the package, so a tenth site --
    or a new trainer copying the idiom back in -- fails here instead of on a
    multi-GPU box nobody develops on."""

    def test_the_scan_actually_sees_the_trainer_package(self):
        """Without this, a moved source tree turns every check below into a
        vacuous pass over an empty file list."""
        assert _TRAINER_DIR.is_dir(), _TRAINER_DIR
        names = {path.name for path in _TRAINER_SOURCES}
        assert len(names) > 20
        # the modules the H100 run proved are reachable with `--gpus`
        assert {"sft.py", "dpo.py", "orpo.py", "simpo.py", "ppo.py"} <= names

    @pytest.mark.parametrize(
        "path", _TRAINER_SOURCES, ids=[p.stem for p in _TRAINER_SOURCES]
    )
    def test_module_does_not_hardcode_device_map_auto(self, path):
        code = _code_without_comments(path.read_text(encoding="utf-8"))
        assert not _OLD_IDIOM.search(code), (
            f"{path.name} still picks device_map by hand; under `accelerate "
            f"launch` transformers raises on 'auto'. Use resolve_device_map()."
        )
        assert not _INLINE_AUTO.search(code), (
            f"{path.name} passes device_map='auto' literally; same defect."
        )

    @pytest.mark.parametrize(
        "path", _TRAINER_SOURCES, ids=[p.stem for p in _TRAINER_SOURCES]
    )
    def test_module_that_sets_a_device_map_uses_the_shared_helper(self, path):
        """The negative check above is satisfied by deleting the line entirely.
        This is its positive half: whatever a module does pass must come from
        the one helper that knows about LOCAL_RANK."""
        code = _code_without_comments(path.read_text(encoding="utf-8"))
        mentions = re.sub(r"hf_device_map|_device_map_value", "", code)
        if "device_map" not in mentions:
            return
        assert "resolve_device_map" in code, (
            f"{path.name} passes a device_map that does not come from "
            "resolve_device_map()"
        )

    def test_the_patterns_would_catch_the_old_lines(self):
        """A scanner nobody has seen fail is not evidence. Both spellings the
        H100 run found, plus a renamed variant, must match -- and the fixed line
        must not."""
        assert _OLD_IDIOM.search('dev_map = "cpu" if self.device == "cpu" else "auto"')
        assert _OLD_IDIOM.search('m = "cpu" if device == "cpu" else "auto"')
        assert _INLINE_AUTO.search('device_map="auto" if self.device != "cpu" else None')
        assert not _OLD_IDIOM.search("dev_map = resolve_device_map(self.device)")
        assert not _INLINE_AUTO.search("device_map=resolve_device_map(self.device)")
