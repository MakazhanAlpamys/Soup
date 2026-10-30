"""Bounded native MLX requests for the official evaluation harness.

The harness owns tasks, prompts, filters and aggregation. MLX-LM owns generation;
this adapter supplies conditional causal likelihood and explicit runtime limits.
All optional imports remain behind native backend selection.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import math
import time
from pathlib import Path
from typing import Any


def validate_options(
    tasks: list[str],
    *,
    batch_size: int,
    limit: int,
    max_length: int,
    max_tokens: int,
    seed: int,
    num_fewshot: int | None,
    device: str | None,
) -> None:
    """Keep the initially qualified task and resource slice explicit."""
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - {"hellaswag", "gsm8k"}:
        raise ValueError("Native MLX benchmark supports hellaswag and gsm8k only")
    if device is not None:
        raise ValueError("MLX selects its native device; omit --device")
    for name, value, low, high in [
        ("batch-size", batch_size, 1, 64),
        ("limit", limit, 1, 1024),
        ("max-length", max_length, 2, 32768),
        ("max-tokens", max_tokens, 1, 16384),
        ("seed", seed, 0, 2**32 - 1),
    ]:
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Native MLX --{name} must be between {low} and {high}")
    if max_tokens >= max_length:
        raise ValueError("Native MLX --max-tokens must be smaller than --max-length")
    if num_fewshot is not None and (type(num_fewshot) is not int or not 0 <= num_fewshot <= 64):
        raise ValueError("Native MLX --fewshot must be between 0 and 64")


def _check_harness() -> str:
    from packaging.version import Version

    try:
        import lm_eval  # noqa: F401

        version = metadata.version("lm-eval")
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise ImportError("Install soup-cli[mlx,eval] with lm-eval>=0.4.12,<0.4.14") from exc
    if not Version("0.4.12") <= Version(version) < Version("0.4.14"):
        raise ValueError("Native MLX benchmark requires lm-eval>=0.4.12,<0.4.14")
    return version


def make_native_harness(
    model_path: str,
    base_model: str | None = None,
    *,
    batch_size: int = 8,
    max_length: int = 2048,
    max_tokens: int = 256,
    trust_remote_code: bool = False,
) -> Any:
    """Construct a lazy TemplateLM using strict native artifacts and raw task prompts."""
    _check_harness()
    import mlx.core as mx
    from lm_eval.api.model import TemplateLM
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler

    from soup_cli.utils.mlx_generation import load_native_model

    model, tokenizer = load_native_model(
        model_path,
        base_model,
        trust_remote_code=trust_remote_code,
    )

    class NativeHarness(TemplateLM):
        def __init__(self) -> None:
            super().__init__()
            self.tokenizer = tokenizer
            self.scored_token_counts: list[int] = []
            self.generation_limits: list[dict] = []

        @property
        def eot_token_id(self) -> int:
            token = tokenizer.eos_token_id
            if type(token) is not int or token < 0:
                raise ValueError("Native benchmark requires a tokenizer EOS token")
            return token

        def tok_encode(
            self,
            string: str,
            add_special_tokens: bool | None = None,
            **kwargs: Any,
        ) -> list[int]:
            if add_special_tokens is None:
                bos = tokenizer.bos_token
                add_special_tokens = not bos or not string.startswith(bos)
            return tokenizer.encode(string, add_special_tokens=add_special_tokens, **kwargs)

        def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
            # TemplateLM moves trailing whitespace into the jointly encoded continuation.
            # Some BPEs also merge a non-whitespace boundary: refuse rather than silently
            # dropping a token or replacing the true context with an independently encoded one.
            context_ids, continuation_ids = super()._encode_pair(context, continuation)
            whole = self.tok_encode(context + continuation)
            if whole[: len(context_ids)] != context_ids:
                raise ValueError("Unsupported tokenizer merge across the continuation boundary")
            return context_ids or [self.prefix_token_id], continuation_ids

        def loglikelihood(self, requests: list, disable_tqdm: bool = False) -> list:
            for request in requests:
                context, continuation = request.args
                if not isinstance(context, str) or not isinstance(continuation, str):
                    raise ValueError("Native likelihood requires text context and continuation")
                if not self.tok_encode(continuation, add_special_tokens=False):
                    raise ValueError("Native likelihood requires a nonempty continuation")
            return super().loglikelihood(requests, disable_tqdm=disable_tqdm)

        def _loglikelihood_tokens(self, requests: list, **kwargs: Any) -> list[tuple[float, bool]]:
            results = []
            for offset in range(0, len(requests), batch_size):
                chunk = requests[offset : offset + batch_size]
                inputs = []
                for _, context, continuation in chunk:
                    if not continuation or len(continuation) > max_length:
                        raise ValueError("Native continuation must contain 1..max-length tokens")
                    inputs.append((context + continuation)[-(max_length + 1) :][:-1])
                width = max(map(len, inputs))
                tokens = mx.array(
                    [row + [self.eot_token_id] * (width - len(row)) for row in inputs]
                )
                logits = model(tokens).astype(mx.float32)
                for index, (key, _, continuation) in enumerate(chunk):
                    end = len(inputs[index])
                    selected = logits[index, end - len(continuation) : end]
                    log_probs = selected - mx.logsumexp(selected, axis=-1, keepdims=True)
                    targets = mx.array(continuation)
                    score = mx.take_along_axis(log_probs, targets[:, None], axis=-1).sum().item()
                    greedy = bool(mx.all(mx.argmax(selected, axis=-1) == targets).item())
                    if not math.isfinite(score):
                        raise ValueError("Native benchmark produced nonfinite likelihood")
                    result = (score, greedy)
                    self.scored_token_counts.append(len(continuation))
                    results.append(result)
                    self.cache_hook.add_partial("loglikelihood", key, result)
            return results

        def loglikelihood_rolling(self, requests: list, **kwargs: Any) -> list:
            raise ValueError("Native MLX rolling loglikelihood is not supported")

        def generate_until(self, requests: list, disable_tqdm: bool = False) -> list[str]:
            results = []
            for offset in range(0, len(requests), batch_size):
                chunk = requests[offset : offset + batch_size]
                prompts, caps, stops = [], [], []
                for request in chunk:
                    prompt, options = request.args
                    if not isinstance(prompt, str) or not isinstance(options, dict):
                        raise ValueError("Native generation requires a text prompt and options")
                    if set(options) - {"until", "max_gen_toks", "do_sample", "temperature"}:
                        raise ValueError("Unsupported native benchmark generation options")
                    if (
                        options.get("do_sample", False) is not False
                        or options.get("temperature", 0) != 0
                    ):
                        raise ValueError("Native benchmark supports greedy generation only")
                    cap = options.get("max_gen_toks", max_tokens)
                    if type(cap) is not int or cap < 1:
                        raise ValueError("Native max_gen_toks must be a positive integer")
                    cap = min(cap, max_tokens)
                    until = options.get("until", [])
                    if isinstance(until, str):
                        until = [until]
                    if not isinstance(until, list) or any(
                        not isinstance(stop, str) or not stop for stop in until
                    ):
                        raise ValueError("Native until must contain nonempty text stops")
                    ids = self.tok_encode(prompt) or [self.prefix_token_id]
                    prompts.append(ids[-(max_length - cap) :])
                    caps.append(cap)
                    stops.append(until)
                    self.generation_limits.append(
                        {
                            "prompt_tokens": len(ids),
                            "retained_prompt_tokens": len(prompts[-1]),
                            "max_tokens": cap,
                            "until": until,
                        }
                    )
                response = batch_generate(
                    model,
                    tokenizer,
                    prompts=prompts,
                    max_tokens=caps,
                    sampler=make_sampler(temp=0),
                    verbose=False,
                )
                if len(response.texts) != len(chunk):
                    raise ValueError("Native generation returned an incomplete batch")
                for request, text, until in zip(chunk, response.texts, stops):
                    if not isinstance(text, str):
                        raise ValueError("Native generation returned nontext output")
                    end = min(
                        (text.find(stop) for stop in until if stop in text), default=len(text)
                    )
                    text = text[:end]
                    results.append(text)
                    self.cache_hook.add_partial("generate_until", request.args, text)
            return results

    return NativeHarness()


def _valid_sample_payload(sample: dict, *, generation: bool) -> bool:
    """Check the two qualified request/response shapes, including valid empty generations."""
    def valid_response(value: Any) -> bool:
        if generation:
            return isinstance(value, str)
        return (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[0], (int, float))
            and not isinstance(value[0], bool)
            and math.isfinite(value[0])
            and type(value[1]) is bool
        )

    return (
        all(
            isinstance(request, (list, tuple))
            and len(request) == 2
            and isinstance(request[0], str)
            and isinstance(request[1], dict if generation else str)
            for request in sample["arguments"]
        )
        and all(
            isinstance(group, (list, tuple))
            and bool(group)
            and all(valid_response(response) for response in group)
            for group in sample["resps"]
        )
        and all(valid_response(response) for response in sample["filtered_resps"])
    )


def _validate_sample_evidence(
    name: str, samples: list[dict], filters: set[str], count: int, metric: str,
) -> None:
    """Require actual requests, responses and matching documents in every filter."""
    by_filter = {key: [] for key in filters}
    identities = {}
    hash_fields = ("doc_hash", "prompt_hash", "target_hash")
    required_fields = {"doc", "target", "arguments", "resps", "filtered_resps", "metrics"}
    for sample in samples:
        if (
            not isinstance(sample, dict)
            or sample.get("filter") not in by_filter
            or type(sample.get("doc_id")) is not int
            or sample["doc_id"] < 0
            or not required_fields.issubset(sample)
            or not isinstance(sample["doc"], dict)
            or any(not isinstance(sample.get(field), str) or not sample[field]
                   for field in hash_fields)
            or any(not isinstance(sample[field], list) or not sample[field]
                   for field in ("arguments", "resps", "filtered_resps", "metrics"))
        ):
            raise ValueError(f"Native benchmark {name} has invalid sample/filter evidence")
        metric_names = sample["metrics"]
        if (
            metric not in metric_names
            or any(not isinstance(metric_name, str)
                   or not isinstance(sample.get(metric_name), (int, float))
                   or isinstance(sample[metric_name], bool)
                   or not math.isfinite(sample[metric_name]) for metric_name in metric_names)
            or len(sample["arguments"]) != len(sample["resps"])
            or len(sample["resps"]) != len(sample["filtered_resps"])
            or not _valid_sample_payload(sample, generation=metric == "exact_match")
        ):
            raise ValueError(f"Native benchmark {name} has invalid sample measurement evidence")
        sample_identity = tuple(sample[field] for field in hash_fields)
        doc_id = sample["doc_id"]
        if doc_id in identities and identities[doc_id] != sample_identity:
            raise ValueError(
                f"Native benchmark {name} has inconsistent sample document evidence"
            )
        identities[doc_id] = sample_identity
        by_filter[sample["filter"]].append(doc_id)
    if len({frozenset(ids) for ids in by_filter.values()}) != 1:
        raise ValueError(f"Native benchmark {name} has disjoint sample/filter evidence")
    complete_filters = all(
        len(ids) == count
        and len(set(ids)) == count
        for ids in by_filter.values()
    )
    if not complete_filters:
        raise ValueError(f"Native benchmark {name} has incomplete sample measurement evidence")


def run_native_benchmark(
    model_path: str,
    tasks: list[str],
    *,
    base_model: str | None,
    num_fewshot: int | None,
    batch_size: int,
    limit: int,
    max_length: int,
    max_tokens: int,
    seed: int,
    trust_remote_code: bool,
) -> dict:
    """Let the official harness measure tasks, then validate all evidence before persistence."""
    version = _check_harness()
    import lm_eval
    import mlx.core as mx
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.utils import handle_non_serializable

    from soup_cli.utils.mlx_generation import read_native_adapter

    started = time.monotonic()
    identity = read_native_adapter(Path(model_path), base_model)
    manager = TaskManager()
    resolved = get_task_dict(tasks, task_manager=manager)
    lm = make_native_harness(
        model_path,
        base_model,
        batch_size=batch_size,
        max_length=max_length,
        max_tokens=max_tokens,
        trust_remote_code=trust_remote_code,
    )
    if set(resolved) != set(tasks) or any(
        getattr(task, "OUTPUT_TYPE", None)
        not in {"multiple_choice", "loglikelihood", "generate_until"}
        for task in resolved.values()
    ):
        raise ValueError("Unsupported native benchmark task/request type (including rolling)")
    results = lm_eval.simple_evaluate(
        model=lm,
        task_manager=manager,
        tasks=list(resolved.values()),
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        bootstrap_iters=100,
        log_samples=True,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
    )
    if not isinstance(results, dict):
        raise ValueError("Native benchmark returned no measured results")
    selected_metrics = {}
    for name in tasks:
        data = results.get("results", {}).get(name, {})
        keys = [
            "acc,none",
            "acc_norm,none",
            "exact_match,strict-match",
            "exact_match,flexible-extract",
            "exact_match,none",
        ]
        key = next((key for key in keys if key in data), None)
        value = data.get(key) if key else None
        measured = (
            (key.split(",")[0], float(value))
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )
        selected_metrics[name] = key
        count = results.get("n-samples", {}).get(name, {}).get("effective", 0)
        samples = results.get("samples", {}).get(name, [])
        config = results.get("configs", {}).get(name, {})
        expected_filters = {item["name"] for item in (config.get("filter_list") or [])} or {"none"}
        if (
            measured is None
            or measured[0] not in {"acc", "acc_norm", "exact_match", "em"}
            or not math.isfinite(measured[1])
            or type(count) is not int
            or count < 1
            or count > limit
            or name not in results.get("versions", {})
            or name not in results.get("configs", {})
        ):
            raise ValueError(
                f"Native benchmark {name} has missing/empty/nonfinite measurement evidence"
            )
        _validate_sample_evidence(name, samples, expected_filters, count, key.split(",")[0])
    results["_soup_metrics"] = {
        name: (key.split(",")[0], float(results["results"][name][key]))
        for name, key in selected_metrics.items()
    }
    artifacts = {}
    for name in ["adapter_config.json", "adapters.safetensors", "config.json", "tokenizer.json"]:
        path = Path(model_path) / name
        if path.is_file():
            artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    results["_soup_native"] = {
        "backend": "mlx",
        "model": model_path,
        "base": identity["model"] if identity else model_path,
        "adapter": model_path if identity else None,
        "adapter_metadata": identity,
        "artifact_sha256": artifacts,
        "versions": {name: metadata.version(name) for name in ["mlx", "mlx-lm", "transformers"]},
        "selected_metrics": selected_metrics,
        "harness_version": version,
        "tasks": tasks,
        "batch_size": batch_size,
        "limit": limit,
        "max_length": max_length,
        "max_tokens": max_tokens,
        "seed": seed,
        "num_fewshot_override": num_fewshot,
        "apply_chat_template": False,
        "bootstrap_iters": 100,
        "trust_remote_code": trust_remote_code,
        "scored_token_counts": lm.scored_token_counts,
        "generation_limits": lm.generation_limits,
        "elapsed_seconds": time.monotonic() - started,
        "process_peak_memory_bytes": mx.get_peak_memory(),
    }
    # Validate serialization before opening the tracker; no partially saved task on bad evidence.
    return json.loads(json.dumps(results, default=handle_non_serializable, allow_nan=False))
