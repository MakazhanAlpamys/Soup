"""Tests for curriculum learning — config, sorting, bucket creation."""

from soup_cli.config.schema import SoupConfig, TrainingConfig

# ─── Config Tests ─────────────────────────────────────────────────────────


class TestCurriculumConfig:
    """Test curriculum learning fields in TrainingConfig."""

    def test_curriculum_default_false(self):
        """curriculum should default to False."""
        tcfg = TrainingConfig()
        assert tcfg.curriculum is False

    def test_curriculum_true(self):
        tcfg = TrainingConfig(curriculum=True)
        assert tcfg.curriculum is True

    def test_curriculum_metric_default(self):
        """curriculum_metric should default to 'length'."""
        tcfg = TrainingConfig()
        assert tcfg.curriculum_metric == "length"

    def test_curriculum_metric_length(self):
        tcfg = TrainingConfig(curriculum_metric="length")
        assert tcfg.curriculum_metric == "length"

    def test_curriculum_metric_perplexity(self):
        tcfg = TrainingConfig(curriculum_metric="perplexity")
        assert tcfg.curriculum_metric == "perplexity"

    def test_curriculum_metric_loss(self):
        tcfg = TrainingConfig(curriculum_metric="loss")
        assert tcfg.curriculum_metric == "loss"

    def test_curriculum_buckets_default(self):
        """curriculum_buckets should default to 4."""
        tcfg = TrainingConfig()
        assert tcfg.curriculum_buckets == 4

    def test_curriculum_buckets_custom(self):
        tcfg = TrainingConfig(curriculum_buckets=8)
        assert tcfg.curriculum_buckets == 8

    def test_curriculum_in_full_config(self):
        cfg = SoupConfig(
            base="test-model",
            data={"train": "data.jsonl"},
            training={
                "curriculum": True,
                "curriculum_metric": "length",
                "curriculum_buckets": 4,
            },
        )
        assert cfg.training.curriculum is True
        assert cfg.training.curriculum_metric == "length"
        assert cfg.training.curriculum_buckets == 4


# ─── YAML Config Loading Tests ────────────────────────────────────────────


class TestCurriculumYamlConfig:
    """Test curriculum via YAML config loading."""

    def test_load_config_with_curriculum(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: test-model
data:
  train: data.jsonl
training:
  curriculum: true
  curriculum_metric: length
  curriculum_buckets: 6
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.training.curriculum is True
        assert cfg.training.curriculum_metric == "length"
        assert cfg.training.curriculum_buckets == 6

    def test_load_config_without_curriculum(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: test-model
data:
  train: data.jsonl
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.training.curriculum is False


# ─── Curriculum Sorting Tests ─────────────────────────────────────────────


class TestCurriculumSorting:
    """Test curriculum sorting by different metrics."""

    def test_sort_by_length(self):
        """Sort by length should order short → long."""
        from soup_cli.utils.curriculum import sort_by_length

        data = [
            {"text": "a" * 100},
            {"text": "a" * 10},
            {"text": "a" * 50},
        ]
        sorted_data = sort_by_length(data)
        lengths = [len(row["text"]) for row in sorted_data]
        assert lengths == [10, 50, 100]

    def test_sort_by_length_messages_format(self):
        """Sort by length should work with messages format."""
        from soup_cli.utils.curriculum import sort_by_length

        data = [
            {"messages": [{"role": "user", "content": "a" * 100}]},
            {"messages": [{"role": "user", "content": "short"}]},
            {"messages": [{"role": "user", "content": "a" * 50}]},
        ]
        sorted_data = sort_by_length(data)
        # First should be shortest
        assert len(str(sorted_data[0])) < len(str(sorted_data[-1]))


# ─── Bucket Creation Tests ───────────────────────────────────────────────


class TestCurriculumBuckets:
    """Test bucket creation for staged training."""

    def test_create_buckets(self):
        """create_buckets should split sorted data into N roughly equal parts."""
        from soup_cli.utils.curriculum import create_buckets

        data = list(range(100))
        buckets = create_buckets(data, num_buckets=4)
        assert len(buckets) == 4
        # All data should be present
        flat = [item for bucket in buckets for item in bucket]
        assert sorted(flat) == data

    def test_create_buckets_uneven(self):
        """Buckets should handle non-evenly-divisible data."""
        from soup_cli.utils.curriculum import create_buckets

        data = list(range(10))
        buckets = create_buckets(data, num_buckets=3)
        assert len(buckets) == 3
        flat = [item for bucket in buckets for item in bucket]
        assert sorted(flat) == data

    def test_create_buckets_single(self):
        """Single bucket should contain all data."""
        from soup_cli.utils.curriculum import create_buckets

        data = list(range(20))
        buckets = create_buckets(data, num_buckets=1)
        assert len(buckets) == 1
        assert buckets[0] == data

    def test_create_buckets_more_than_data(self):
        """More buckets than data should still work."""
        from soup_cli.utils.curriculum import create_buckets

        data = list(range(3))
        buckets = create_buckets(data, num_buckets=5)
        # Some buckets may be empty, but all data should be present
        flat = [item for bucket in buckets for item in bucket]
        assert sorted(flat) == data


# ─── Sweep Integration Tests ─────────────────────────────────────────────


class TestCurriculumSweep:
    """Test curriculum in sweep configurations."""

    def test_curriculum_in_sweep_params(self):
        from soup_cli.commands.sweep import _parse_sweep_params

        params = _parse_sweep_params(["training.curriculum=true,false"])
        assert "training.curriculum" in params

    def test_curriculum_buckets_in_sweep_params(self):
        from soup_cli.commands.sweep import _parse_sweep_params

        params = _parse_sweep_params(["training.curriculum_buckets=2,4,8"])
        assert "training.curriculum_buckets" in params
        assert params["training.curriculum_buckets"] == [2, 4, 8]
