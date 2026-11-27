"""Cache history must amortize setup, respect ceilings and fail safely when stale."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.stream_cache_profile import (
    CacheMeasurement,
    choose_cache,
    load_measurements,
    profile_identity,
    save_measurement,
)


def rows():
    return [CacheMeasurement(0, 0, 20., 10.), CacheMeasurement(
        14336, 2, 40., 9., ram_indices=(0, 3), gpu_indices=(1, 2),
        baseline_allocated=100, baseline_reserved=120,
    )]


def test_short_job_avoids_setup_and_long_job_reuses_amortized_cache():
    assert choose_cache(rows(), 18, ram_mib=14336, resident_limit="auto").ordinary
    assert choose_cache(rows(), 100, ram_mib=14336, resident_limit="auto") == rows()[1]
    assert choose_cache(rows(), 100, ram_mib=8192, resident_limit="auto").ordinary
    assert choose_cache(rows(), 100, ram_mib=14336, resident_limit=1).ordinary
    assert choose_cache(rows()[1:], 100, ram_mib=14336, resident_limit="auto") is None


def test_reuse_excludes_record_probe_time_from_the_estimated_job(tmp_path):
    baseline, weights = rows()
    assert choose_cache([baseline, weights], 18, ram_mib=14336, resident_limit="auto").ordinary

    measured = replace(weights, record_only_seconds=10.)
    path = tmp_path / "cache.json"
    for candidate in (baseline, measured):
        save_measurement(path, "key", candidate)
    history = load_measurements(path, "key")
    assert history == [baseline, measured]
    assert measured.startup_seconds == weights.startup_seconds
    assert choose_cache(history, 18, ram_mib=14336, resident_limit="auto") == measured
    assert choose_cache(history, 4, ram_mib=14336, resident_limit="auto").ordinary


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, 41])
def test_record_probe_time_must_be_within_the_measured_startup(value):
    with pytest.raises(ValueError, match="record-only time"):
        replace(rows()[1], record_only_seconds=value)


def test_close_call_keeps_ordinary_streaming():
    candidates = [CacheMeasurement(0, 0, 0, 10), CacheMeasurement(1024, 0, 0, 9.9)]
    assert choose_cache(candidates, 100, ram_mib=1024, resident_limit=0).ordinary


def test_joint_placement_compares_weight_and_activation_use_of_memory(tmp_path):
    ordinary, weights = rows()
    activations = CacheMeasurement(0, 0, 30., 9., retained_layers=3)
    both = replace(weights, startup_seconds=100., step_seconds=7., retained_layers=2)
    path = tmp_path / "cache.json"
    for candidate in (ordinary, weights, activations, both):
        save_measurement(path, "key", candidate)
    measured = load_measurements(path, "key")
    assert len(measured) == 4
    limits = dict(ram_mib=14336, resident_limit=2)
    assert choose_cache(measured, 4, **limits, allow_retention=True) == ordinary
    assert choose_cache(measured, 18, **limits, allow_retention=True) == activations
    assert choose_cache(measured, 100, **limits, allow_retention=True) == both
    assert choose_cache(measured, 100, **limits) == weights
    assert choose_cache(measured, 100, ram_mib=0, resident_limit=0,
                        allow_retention=True) == activations


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_unusable_timing_refuses(value):
    with pytest.raises(ValueError):
        CacheMeasurement(0, 0, 0, value)


def test_saved_profiles_round_trip_and_reject_other_workloads(tmp_path):
    path = tmp_path / "cache.json"
    key = profile_identity({"gpu": "uuid-a", "model": "weights-a", "seq": 512})
    for row in rows():
        save_measurement(path, key, row)
    assert load_measurements(path, key) == rows()
    changed = profile_identity({"gpu": "uuid-a", "model": "weights-b", "seq": 512})
    assert not load_measurements(path, changed)
    save_measurement(path, key, CacheMeasurement(0, 0, 21, 10))
    assert len(load_measurements(path, key)) == 2
    assert load_measurements(path, key)[1].startup_seconds == 21


def test_stale_and_corrupt_profiles_cannot_grant_a_cache(tmp_path, monkeypatch):
    import soup_cli.utils.stream_cache_profile as module

    path = tmp_path / "cache.json"
    save_measurement(path, "key", rows()[0])
    now = module.time.time()
    monkeypatch.setattr(module.time, "time", lambda: now + module.PROFILE_MAX_AGE + 1)
    assert not load_measurements(path, "key")
    path.write_text('{"version":1,"profiles":{"key":{"updated":0,"rows":"bad"}}}')
    monkeypatch.setattr(module.time, "time", lambda: 0)
    with pytest.raises(ValueError):
        load_measurements(path, "key")


def test_concurrent_writer_leaves_complete_profile_untouched(tmp_path):
    path = tmp_path / "cache.json"
    save_measurement(path, "key", rows()[0])
    before = path.read_bytes()
    path.with_name("cache.json.lock").write_text("busy")
    with pytest.raises(FileExistsError):
        save_measurement(path, "key", rows()[1])
    assert path.read_bytes() == before


def test_profile_identity_binds_runtime_data_and_math_but_not_job_length(tmp_path, monkeypatch):
    import torch

    import soup_cli.utils.stream_cache_profile as module

    index = tmp_path / "index.json"
    index.write_text('{"model":"a"}')
    cfg = SoupConfig.model_validate(payload())
    dataset = SimpleNamespace(_fingerprint="tokens-a")
    owner = SimpleNamespace(trainer=SimpleNamespace(train_dataset=dataset),
                            _stream_runtime=SimpleNamespace(pool=SimpleNamespace(device="cpu")))
    args = SimpleNamespace(world_size=1, per_device_train_batch_size=1,
                           gradient_accumulation_steps=2)
    optimizer = SimpleNamespace(defaults={"lr": .002, "foreach": False})
    properties = SimpleNamespace(name="card", uuid="GPU-test", total_memory=8 * 2**30,
                                 major=8, minor=9)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_: properties)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *_a, **_k: "590.00")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "1.0")
    key = module._runtime_identity(owner, cfg, args, index, optimizer)
    properties.uuid = "test"
    assert module._runtime_identity(owner, cfg, args, index, optimizer) == key
    cfg.training.epochs += 1
    cfg.training.stream_ram_cache_mb = 1024
    cfg.training.stream_resident_layers = "auto"
    cfg.training.stream_cache_profile_mode = "record"
    cfg.training.stream_recompute = "auto"
    assert module._runtime_identity(owner, cfg, args, index, optimizer) == key
    dataset._fingerprint = "tokens-b"
    assert module._runtime_identity(owner, cfg, args, index, optimizer) != key
    dataset._fingerprint = "tokens-a"
    optimizer.defaults["foreach"] = True
    assert module._runtime_identity(owner, cfg, args, index, optimizer) != key


@pytest.mark.parametrize("raw", ['[]', '{"version":1,"profiles":{"x":0}}'])
def test_malformed_profile_shape_refuses(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_measurements(path, "x")


def payload(**extra):
    return dict(base="local-model", task="sft", data={"train": "x.jsonl"}, training=dict(
        stream_layers=True, stream_source="disk", batch_size=1,
        stream_cache_profile="cache.json", **extra,
    ))


def test_schema_default_and_record_mode():
    assert SoupConfig.model_validate(payload()).training.stream_cache_profile_mode == "reuse"
    assert SoupConfig.model_validate(payload(
        stream_cache_profile_mode="record",
    )).training.stream_cache_profile_mode == "record"
    default = SoupConfig.model_validate({"base": "local", "data": {"train": "x.jsonl"}})
    assert default.training.stream_cache_profile is None


@pytest.mark.parametrize("field,value", [
    ("stream_source", "ram"), ("stream_expert_layout", "per_expert"),
    ("batch_size", "auto"), ("stream_vram_probe", True), ("activation_offloading", "cpu"),
])
def test_profile_scope_refuses_unmeasured_paths(field, value):
    value_dict = payload()
    value_dict["training"][field] = value
    with pytest.raises(ValueError, match="stream_cache_profile"):
        SoupConfig.model_validate(value_dict)


def test_yaml_on_records_with_checkpoint_recomputation():
    from soup_cli.config.loader import load_config_from_string

    cfg = load_config_from_string(
        "base: local-model\ndata: {train: x.jsonl}\ntraining:\n"
        "  stream_layers: true\n  stream_source: disk\n  batch_size: 1\n"
        "  stream_cache_profile: profile.json\n  stream_recompute: on\n"
    )
    assert cfg.training.stream_recompute == "on"


def test_profile_path_refuses_url_and_streaming_off():
    value = payload()
    value["training"]["stream_cache_profile"] = "https://invalid/profile"
    with pytest.raises(ValueError, match="stream_cache_profile"):
        SoupConfig.model_validate(value)
    value = payload()
    value["training"]["stream_layers"] = False
    with pytest.raises(ValueError, match="stream_cache_profile"):
        SoupConfig.model_validate(value)


def test_retention_only_profile_requires_a_budgetable_optimizer():
    with pytest.raises(ValueError, match="requires an AdamW optimizer and fast_lora=false"):
        SoupConfig.model_validate(payload(stream_recompute="auto", optimizer="adafactor"))


def test_short_reuse_never_probes_or_times_updates(tmp_path, monkeypatch):
    import soup_cli.utils.stream_cache_profile as module

    path = tmp_path / "cache.json"
    for row in rows():
        save_measurement(path, "key", row)
    monkeypatch.setattr(module, "_runtime_identity", lambda *_: "key")
    cfg = SoupConfig.model_validate(payload(stream_ram_cache_mb=14336,
                                          stream_resident_layers="auto"))
    cfg.training.stream_cache_profile = str(path)
    runtime = SimpleNamespace(tier="disk", pool=SimpleNamespace(is_cuda=True))
    owner = SimpleNamespace(_stream_runtime=runtime)
    messages = []
    callback = module.build_stream_cache_callback(
        owner, cfg, index_path=tmp_path / "index.json", setup_started=0, notify=messages.append,
    )
    callback.on_train_begin(None, SimpleNamespace(max_steps=18), None,
                            model=object(), optimizer=object())
    assert not callback.recording
    callback.on_step_begin(None, None, None)
    callback.on_step_end(None, None, None)
    assert "ordinary streaming" in messages[-1]


def test_profile_defers_all_cache_allocation_until_trainer_knows_steps(tmp_path, monkeypatch):
    import soup_cli.utils.stream_cache_profile as module
    import soup_cli.utils.stream_residency as residency
    from tests.test_issue623_stream_pin_allocation import _drive_wiring

    calls = []
    monkeypatch.setattr(module, "build_stream_cache_callback", lambda *args, **kw: calls.append(kw))
    monkeypatch.setattr(residency, "configure_stream_residency", lambda *_a, **_k: pytest.fail(
        "profile must not pay the eager residency probes",
    ))
    monkeypatch.setattr(
        "soup_cli.utils.stream_cache_recompute.configure_cached_stream_recompute",
        lambda *_a, **_k: pytest.fail("profile must defer activation probes too"),
    )
    captured, _ = _drive_wiring(
        tmp_path, monkeypatch, stream_pin_yaml="", stream_source="disk",
        extra_training_yaml="  stream_cache_profile: cache.json\n"
                            "  stream_resident_layers: auto\n  stream_ram_cache_mb: 1024\n"
                            "  stream_recompute: auto\n",
    )
    assert not captured.get("ram_cache_bytes")
    assert len(calls) == 1 and calls[0]["setup_started"] > 0


@pytest.mark.parametrize("retained", [None, 2, 0])
def test_recording_uses_real_updates_without_replay(tmp_path, monkeypatch, retained):
    import torch

    import soup_cli.utils.stream_cache_profile as module

    path = tmp_path / "cache.json"
    cfg = SoupConfig.model_validate(payload(
        stream_cache_profile_mode="record", stream_recompute="on" if retained is None else "auto",
    ))
    cfg.training.stream_cache_profile = str(path)
    monkeypatch.setattr(module, "_runtime_identity", lambda *_: "key")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    retention_calls = []

    def retention(*_args, **kwargs):
        retention_calls.append(kwargs["retain"])
        return SimpleNamespace(retained_layers=retained, record_only_seconds=3.0)

    monkeypatch.setattr(
        "soup_cli.utils.stream_cache_recompute.configure_cached_stream_recompute", retention,
    )
    if retained == 0:
        save_measurement(path, "key", rows()[0])
    runtime = SimpleNamespace(tier="disk", pool=SimpleNamespace(
        is_cuda=True, device="cpu", resident_slots={},
    ), source=SimpleNamespace(ram_cache_stats=lambda: {"indices": []}))
    owner = SimpleNamespace(_stream_runtime=runtime, _stream_step_shape=lambda *_: (1, 8, 16))
    clock = iter([20, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock))
    callback = module.build_stream_cache_callback(
        owner, cfg, index_path=tmp_path / "index.json", setup_started=10, notify=lambda _: None,
    )
    callback.on_train_begin(None, SimpleNamespace(max_steps=100), None,
                            model=SimpleNamespace(config=object()), optimizer=object())
    for _ in range(8):
        callback.on_step_begin(None, None, None)
        callback.on_step_end(None, None, None)
    result = load_measurements(path, "key")
    assert retention_calls == ([] if retained is None else [None])
    if retained == 0:
        assert result == [rows()[0]]
    else:
        assert len(result) == 1 and result[0].step_seconds == 1
        assert result[0].startup_seconds == 11
        assert result[0].record_only_seconds == (3.0 if retained else 0.0)
        assert result[0].retained_layers == (retained or 0)
    assert not callback.recording
    # No further synchronization, probes or clock reads after the eight samples.
    callback.on_step_begin(None, None, None)
    callback.on_step_end(None, None, None)


@pytest.mark.parametrize(
    "admission", ["accepted", "gpu_refused", "ram_changed", "retention_refused"],
)
def test_long_reuse_requires_fresh_admission_and_same_membership(tmp_path, monkeypatch, admission):
    import torch

    import soup_cli.utils.stream_cache_profile as module
    import soup_cli.utils.stream_residency as residency

    path = tmp_path / "profile.json"
    ordinary, weights = rows()
    save_measurement(path, "key", ordinary)
    save_measurement(path, "key", replace(weights, retained_layers=3))
    cfg = SoupConfig.model_validate(payload(stream_ram_cache_mb=14336,
                                          stream_resident_layers="auto", stream_recompute="auto"))
    cfg.training.stream_cache_profile = str(path)
    monkeypatch.setattr(module, "_runtime_identity", lambda *_: "key")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    calls, cleared = [], []
    host_indices = [0, 3] if admission != "ram_changed" else [0]
    runtime = SimpleNamespace(tier="disk", n_layers=4, pool=SimpleNamespace(
        is_cuda=True, device="cuda:0", resident_slots={1: None, 2: None},
    ), source=SimpleNamespace(configure_ram_cache=calls.append,
                              ram_cache_stats=lambda: {"indices": host_indices}))
    owner = SimpleNamespace(_stream_runtime=runtime, _stream_step_shape=lambda *_: (1, 8, 16))
    probes = []

    def probe(_model, _runtime, mode, **kwargs):
        assert mode == "auto"
        assert kwargs["hint"] == residency.ResidencyHint(2, 100, 120)
        probes.append(kwargs)
        return SimpleNamespace(enabled=admission != "gpu_refused")

    def retention(_model, _runtime, **kwargs):
        assert calls == [14336 * 2**20] and len(probes) == 1
        assert kwargs["retain"] == 3
        return SimpleNamespace(retained_layers=0 if admission == "retention_refused" else 3)

    monkeypatch.setattr(residency, "configure_stream_residency", probe)
    monkeypatch.setattr(
        "soup_cli.utils.stream_cache_recompute.configure_cached_stream_recompute", retention,
    )
    monkeypatch.setattr(module, "_clear_profile_caches", lambda *_: cleared.append(True))
    messages = []
    callback = module.build_stream_cache_callback(
        owner, cfg, index_path=tmp_path / "index.json", setup_started=0, notify=messages.append,
    )
    callback.on_train_begin(None, SimpleNamespace(max_steps=100), None,
                            model=SimpleNamespace(config=object()), optimizer=object())
    assert len(probes) == 1
    assert calls == ([] if admission == "gpu_refused" else [14336 * 2**20])
    assert cleared == ([True] if admission in ("ram_changed", "retention_refused") else [])
    assert hasattr(owner, "_stream_cache_profile_selection") == (admission == "accepted")
    if admission == "accepted":
        assert owner._stream_cache_profile_selection["retained_layers"] == 3
    assert not callback.recording


@pytest.mark.parametrize("invalid", ["gpu_indices", "retained_layers"])
def test_corrupt_profile_indices_keep_ordinary_streaming_without_a_probe(
    tmp_path, monkeypatch, invalid,
):
    import soup_cli.utils.stream_cache_profile as module
    import soup_cli.utils.stream_residency as residency

    path = tmp_path / "profile.json"
    save_measurement(path, "key", rows()[0])
    changes = {invalid: (10, 11) if invalid == "gpu_indices" else 5}
    save_measurement(path, "key", replace(rows()[1], **changes))
    cfg = SoupConfig.model_validate(payload(stream_ram_cache_mb=14336,
                                          stream_resident_layers="auto", stream_recompute="auto"))
    cfg.training.stream_cache_profile = str(path)
    monkeypatch.setattr(module, "_runtime_identity", lambda *_: "key")
    monkeypatch.setattr(residency, "configure_stream_residency", lambda *_a, **_k: pytest.fail(
        "a corrupt history must be rejected before probing or allocation",
    ))
    owner = SimpleNamespace(_stream_runtime=SimpleNamespace(
        tier="disk", n_layers=4, pool=SimpleNamespace(is_cuda=True),
    ))
    messages = []
    callback = module.build_stream_cache_callback(
        owner, cfg, index_path=tmp_path / "index.json", setup_started=0, notify=messages.append,
    )
    callback.on_train_begin(None, SimpleNamespace(max_steps=100), None,
                            model=object(), optimizer=object())
    assert "invalid layer indices" in messages[-1]
    assert not callback.recording


def _message(exc_info):
    return " ".join(str(exc_info.value).split())


def test_mode_without_a_profile_names_both_fields():
    value = payload(stream_cache_profile_mode="record")
    del value["training"]["stream_cache_profile"]
    with pytest.raises(ValueError) as exc_info:
        SoupConfig.model_validate(value)
    assert (
        "training.stream_cache_profile_mode='record' requires training.stream_cache_profile"
        in _message(exc_info)
    )


def test_profile_fields_without_streaming_name_the_gate():
    value = payload(stream_cache_profile_mode="record")
    value["training"]["stream_layers"] = False
    with pytest.raises(ValueError) as exc_info:
        SoupConfig.model_validate(value)
    assert (
        "training.stream_cache_profile / training.stream_cache_profile_mode set but "
        "stream_layers is false" in _message(exc_info)
    )


def test_profile_beside_recompute_off_is_refused():
    with pytest.raises(ValueError) as exc_info:
        SoupConfig.model_validate(payload(stream_recompute="off"))
    assert (
        "training.stream_cache_profile cannot be combined with "
        "training.stream_recompute='off'" in _message(exc_info)
    )


def test_profile_path_must_stay_under_home_cwd_or_temp(tmp_path, monkeypatch):
    import soup_cli.utils.stream_cache_profile as module

    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    value = payload()
    value["training"]["stream_cache_profile"] = str(tmp_path / "elsewhere" / "p.json")
    with pytest.raises(ValueError, match="must stay under the home, working or temp directory"):
        SoupConfig.model_validate(value)
    value["training"]["stream_cache_profile"] = "../escape.json"
    with pytest.raises(ValueError, match="must stay under the home, working or temp directory"):
        SoupConfig.model_validate(value)
    for inside in (work / "p.json", tmp_path / "home" / "p.json", tmp_path / "tmp" / "p.json"):
        value["training"]["stream_cache_profile"] = str(inside)
        assert SoupConfig.model_validate(value).training.stream_cache_profile == str(inside)


def test_profile_file_symlink_is_refused(tmp_path):
    import os

    real_file = tmp_path / "real.json"
    real_file.write_text("{}")
    file_link = tmp_path / "profile.json"
    try:
        os.symlink(real_file, file_link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="must not be a symlink"):
        load_measurements(file_link, "key")


def test_resident_layers_beside_auto_recompute_is_allowed_only_with_a_profile():
    value = payload(stream_resident_layers="auto", stream_recompute="auto")
    assert SoupConfig.model_validate(value).training.stream_recompute == "auto"
    del value["training"]["stream_cache_profile"]
    with pytest.raises(ValueError) as exc_info:
        SoupConfig.model_validate(value)
    assert (
        "training.stream_resident_layers cannot be combined with "
        "training.stream_recompute='auto'" in _message(exc_info)
    )


def test_resident_layers_beside_recompute_off_stays_refused_with_a_profile():
    with pytest.raises(ValueError) as exc_info:
        SoupConfig.model_validate(payload(stream_resident_layers=2, stream_recompute="off"))
    assert (
        "training.stream_resident_layers cannot be combined with "
        "training.stream_recompute='off'" in _message(exc_info)
    )


@pytest.mark.parametrize("raw, reason", [
    ("{not json", "cache profile"),
    ('{"version":2,"profiles":{}}', "unsupported format"),
    ('{"version":1,"profiles":{"k":{"updated":1,"rows":[{"ram_mib":0}]}}}',
     "invalid cache profile candidate"),
    ('{"version":1,"profiles":{"k":{"updated":1,"rows":[5]}}}',
     "invalid cache profile candidate"),
])
def test_malformed_profile_file_is_refused_with_the_file_named(tmp_path, raw, reason):
    from soup_cli.utils.stream_cache_profile import validate_profile_file

    path = tmp_path / "named-profile.json"
    path.write_text(raw)
    with pytest.raises(ValueError) as exc_info:
        validate_profile_file(path)
    assert "named-profile.json" in str(exc_info.value)
    assert reason in str(exc_info.value)


def test_oversize_profile_file_is_refused_with_the_file_named(tmp_path):
    import soup_cli.utils.stream_cache_profile as module
    from soup_cli.utils.stream_cache_profile import validate_profile_file

    path = tmp_path / "big.json"
    path.write_bytes(b" " * (module.PROFILE_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="big.json: exceeds the size limit"):
        validate_profile_file(path)


def test_missing_and_valid_profile_files_pass_validation(tmp_path):
    from soup_cli.utils.stream_cache_profile import validate_profile_file

    validate_profile_file(tmp_path / "absent.json")
    path = tmp_path / "ok.json"
    save_measurement(path, "key", rows()[0])
    validate_profile_file(path)


def test_reuse_with_a_malformed_history_keeps_ordinary_streaming_and_names_the_file(
    tmp_path, monkeypatch,
):
    import soup_cli.utils.stream_cache_profile as module

    path = tmp_path / "broken-history.json"
    path.write_text("{not json")
    monkeypatch.setattr(module, "_runtime_identity", lambda *_: "key")
    cfg = SoupConfig.model_validate(payload(stream_ram_cache_mb=1024))
    cfg.training.stream_cache_profile = str(path)
    owner = SimpleNamespace(_stream_runtime=SimpleNamespace(
        tier="disk", n_layers=4, pool=SimpleNamespace(is_cuda=True),
    ))
    messages = []
    callback = module.build_stream_cache_callback(
        owner, cfg, index_path=tmp_path / "index.json", setup_started=0, notify=messages.append,
    )
    callback.on_train_begin(None, SimpleNamespace(max_steps=100), None,
                            model=object(), optimizer=object())
    assert "invalid history" in messages[-1] and "broken-history.json" in messages[-1]
    assert not callback.recording
