"""Unsloth notebook (.ipynb) → Soup config migration.

Uses AST parsing only — never exec/eval on notebook code.
"""

import ast
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Trainer class → Soup task mapping. #1214: "cpo" is provisional; CPOTrainer
# migrates to simpo only with loss_type="simpo" and is refused otherwise.
_TRAINER_MAP = {
    "SFTTrainer": "sft",
    "DPOTrainer": "dpo",
    "GRPOTrainer": "grpo",
    "KTOTrainer": "kto",
    "ORPOTrainer": "orpo",
    "PPOTrainer": "ppo",
    "RewardTrainer": "reward_model",
    "BCOTrainer": "bco",
    "OnlineDPOTrainer": "online_dpo",
    "CPOTrainer": "cpo",
}

# Trainer config class → Soup task mapping (and the hyperparameter table: a
# call to any of these is read like TrainingArguments).
_CONFIG_MAP = {
    "SFTConfig": "sft",
    "DPOConfig": "dpo",
    "GRPOConfig": "grpo",
    "KTOConfig": "kto",
    "ORPOConfig": "orpo",
    "PPOConfig": "ppo",
    "RewardConfig": "reward_model",
    "BCOConfig": "bco",
    "OnlineDPOConfig": "online_dpo",
    "CPOConfig": "cpo",
}

# Task → default data format
_TASK_FORMAT_MAP = {
    "sft": "auto",
    "dpo": "dpo",
    "grpo": "auto",
    "kto": "kto",
    "orpo": "dpo",
    "simpo": "dpo",
    "ppo": "auto",
    "reward_model": "dpo",
    "bco": "dpo",
    "online_dpo": "auto",
}


def _parses(source: str) -> bool:
    # Only the real parse reports an invalid escape in the notebook; the probes
    # would repeat each SyntaxWarning with a cell-local line number (#1583 review).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            ast.parse(source)
        except SyntaxError:
            return False
    return True


# Cell magics whose body is Python that IPython goes on to execute; the header goes,
# the body stays. Any other cell magic (%%bash, %%writefile, %%html, ...) is not Python.
_PYTHON_BODY_CELL_MAGICS = ("%%capture", "%%time", "%%timeit", "%%prun")


def _blank(line: str) -> str:
    return "\n" if line.endswith("\n") else ""


def _neutralise_magic(line: str, prefix: str) -> Optional[str]:
    """What replaces a ``!`` / ``%`` line at statement level, or ``None`` when it is not
    one. After complete statements the line is blanked. After a header waiting for its
    suite (``if x:``, or nested headers) it becomes a ``pass`` at the magic's own
    indentation, which is what makes the probe parse, so a suite whose only statements
    were magics stays a valid block. Inside an open bracket or after a backslash neither
    parses, so the line is Python and stays."""
    if _parses(prefix):
        return _blank(line)
    indent = line[: len(line) - len(line.lstrip())]
    if _parses(prefix + indent + "pass\n"):
        return indent + "pass" + _blank(line)
    return None


def _strip_ipython_magics(cell_source: str) -> str:
    """Blank the IPython-only lines of one cell so the Python in it parses (#1579).

    A cell that already parses is returned unchanged. A ``%%`` cell magic whose body
    IPython executes as Python (``%%capture``, ``%%time``, ...) loses only its header
    line; any other cell magic is not Python and is blanked whole. A line whose first
    non-blank character is ``!`` or ``%`` is a magic only at statement level, decided by
    whether the lines before it already form complete statements (a complete header
    such as ``if x:`` counts, and the magic under it becomes an indented ``pass`` so the
    block stays valid), so a ``%`` operator or a ``!=`` at the start of a continuation
    line is left alone. Replaced lines keep their newline, so line numbers (which the
    source-order logic relies on) do not move.
    """
    if _parses(cell_source):
        return cell_source
    lines = cell_source.splitlines(keepends=True)
    first_index = next((i for i, line in enumerate(lines) if line.strip()), None)
    out: list[str] = []
    if first_index is not None and lines[first_index].lstrip().startswith("%%"):
        header = lines[first_index].lstrip()
        if not header.startswith(_PYTHON_BODY_CELL_MAGICS):
            return "".join(_blank(line) for line in lines)
        out = [_blank(line) for line in lines[: first_index + 1]]
        lines = lines[first_index + 1 :]
    continued = False
    for line in lines:
        if continued:
            # IPython joins a magic line ending in a backslash with the next line
            # (18 of the 82 public Unsloth GRPO notebooks install that way).
            out.append(_blank(line))
            continued = line.rstrip("\r\n").endswith("\\")
            continue
        replacement = None
        if line.lstrip().startswith(("!", "%")):
            # ponytail: re-parsing the prefix per magic line is O(n^2) in a cell's
            # lines; notebook cells are short. Track bracket depth if that ever matters.
            replacement = _neutralise_magic(line, "".join(out))
            continued = replacement is not None and line.rstrip("\r\n").endswith("\\")
        out.append(line if replacement is None else replacement)
    return "".join(out)


def migrate_unsloth(notebook_path: Path) -> Dict[str, Any]:
    """Parse an Unsloth .ipynb notebook and return a Soup config dict.

    Extracts parameters from function calls using AST parsing only.
    Returns a dict suitable for config_to_yaml(). Includes a ``_warnings``
    key with a list of human-readable migration notes.
    """
    try:
        raw_text = notebook_path.read_text(encoding="utf-8")
        notebook = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in notebook file: {exc}")

    cells = notebook.get("cells", [])
    code_cells = [
        cell for cell in cells
        if cell.get("cell_type") == "code"
    ]

    if not code_cells:
        raise ValueError("No code cells found in notebook")

    # Combine all code cell sources for AST parsing, remembering where each cell
    # starts so a syntax error can name its cell and line (#1579).
    all_source = ""
    cell_starts: list[int] = []
    for cell in code_cells:
        source = cell.get("source", [])
        if isinstance(source, list):
            source = "".join(source)
        cell_starts.append(all_source.count("\n") + 1)
        all_source += _strip_ipython_magics(source) + "\n"

    # Parse AST - safe, no execution
    try:
        tree = ast.parse(all_source)
    except SyntaxError as exc:
        lineno = exc.lineno or 1
        cell_index = max(i for i, start in enumerate(cell_starts) if start <= lineno)
        raise ValueError(
            f"Could not parse notebook code: {exc.msg} (code cell {cell_index + 1}, "
            f"line {lineno - cell_starts[cell_index] + 1})"
        ) from exc

    warnings: List[str] = []
    base: Optional[str] = None
    max_seq_length: Optional[int] = None
    load_in_4bit: Optional[bool] = None
    load_in_8bit: Optional[bool] = None
    load_in_16bit: Optional[bool] = None
    full_finetuning: Optional[bool] = None
    lora_params: Dict[str, Any] = {}
    training_params: Dict[str, Any] = {}
    task = "sft"
    cpo_loss_type: Optional[str] = None
    output_dir = "./output"

    # Collect variable assignments from notebook code (module-level only, in source order)
    assignments: Dict[str, Any] = {}
    ambiguous: set = set()
    # name -> every module-level binding of it, in source order, so a trainer call can
    # be resolved to the latest binding that precedes it (cells run top to bottom). Any
    # value counts, not only calls: a later `cfg = something_else` must shadow an earlier
    # `cfg = CPOConfig(...)`, and then the loss cannot be read.
    name_bindings: Dict[str, List[ast.AST]] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        val = _ast_to_value(stmt.value)
        for target in stmt.targets:
            if not isinstance(target, ast.Name):
                continue
            name_bindings.setdefault(target.id, []).append(stmt.value)
            if val is _SENTINEL or assignments.get(target.id, val) != val:
                ambiguous.add(target.id)
            else:
                assignments[target.id] = val
    for name in ambiguous:
        assignments.pop(name, None)

    # Visit every call in source order. ast.walk is breadth-first (by nesting depth),
    # which let a module-level `cfg = GRPOConfig(...)` be visited after an inline
    # `SFTTrainer(args=SFTConfig(...))` written earlier, so the SFT warm-up stage won
    # (#1560 review): the shape of Unsloth's GRPO notebooks.
    calls = sorted(
        (n for n in ast.walk(tree) if isinstance(n, ast.Call)),
        key=lambda n: (n.lineno, n.col_offset),
    )
    for node in calls:

        func_name = _get_func_name(node)
        if func_name is None:
            continue

        if func_name == "from_pretrained":
            # FastLanguageModel.from_pretrained(...)
            unresolved: List[tuple[str, str]] = []
            kwargs = _extract_kwargs(node, scope=assignments, unresolved_names=unresolved)
            for kw_arg, var_name in unresolved:
                if kw_arg in ("load_in_4bit", "load_in_8bit", "load_in_16bit", "full_finetuning"):
                    warnings.append(
                        f"Could not read precision argument '{kw_arg}' "
                        f"(passed as variable '{var_name}')."
                    )
            if node.args:
                arg_val = _ast_to_value(node.args[0])
                if arg_val is not _SENTINEL and isinstance(arg_val, str):
                    base = arg_val
            if "model_name" in kwargs:
                base = kwargs["model_name"]
            if "max_seq_length" in kwargs:
                max_seq_length = kwargs["max_seq_length"]
            if "load_in_4bit" in kwargs:
                load_in_4bit = kwargs["load_in_4bit"]
            if "load_in_8bit" in kwargs:
                load_in_8bit = kwargs["load_in_8bit"]
            if "load_in_16bit" in kwargs:
                load_in_16bit = kwargs["load_in_16bit"]
            if "full_finetuning" in kwargs:
                full_finetuning = kwargs["full_finetuning"]

        elif func_name == "get_peft_model":
            # FastLanguageModel.get_peft_model(...)
            kwargs = _extract_kwargs(node, scope=assignments)
            if "r" in kwargs:
                lora_params["r"] = kwargs["r"]
            if "lora_alpha" in kwargs:
                lora_params["alpha"] = kwargs["lora_alpha"]
            if "lora_dropout" in kwargs:
                lora_params["dropout"] = kwargs["lora_dropout"]
            if "target_modules" in kwargs:
                lora_params["target_modules"] = kwargs["target_modules"]
            if "use_dora" in kwargs:
                lora_params["use_dora"] = kwargs["use_dora"]
            if "use_rslora" in kwargs and kwargs["use_rslora"]:
                lora_params["use_rslora"] = True

        elif func_name in _TRAINER_MAP:
            # SFTTrainer(...), DPOTrainer(...), etc.
            task = _TRAINER_MAP[func_name]
            kwargs = _extract_kwargs(node, scope=assignments)
            if func_name == "CPOTrainer":
                cpo_loss_type = _cpo_loss_type(node, kwargs, assignments, name_bindings)
            if kwargs.get("packing"):
                warnings.append(
                    "packing=True is not supported in Soup. "
                    "Sequences will be padded individually."
                )

        elif func_name == "TrainingArguments" or func_name in _CONFIG_MAP:
            # TrainingArguments(...), DPOConfig(...), etc.
            if func_name in _CONFIG_MAP:
                task = _CONFIG_MAP[func_name]
            kwargs = _extract_kwargs(node, scope=assignments)
            if "per_device_train_batch_size" in kwargs:
                training_params["batch_size"] = kwargs["per_device_train_batch_size"]
            if "num_train_epochs" in kwargs:
                training_params["epochs"] = kwargs["num_train_epochs"]
            if "learning_rate" in kwargs:
                training_params["lr"] = kwargs["learning_rate"]
            if "optim" in kwargs:
                training_params["optimizer"] = kwargs["optim"]
            if "lr_scheduler_type" in kwargs:
                training_params["scheduler"] = kwargs["lr_scheduler_type"]
            if "output_dir" in kwargs:
                output_dir = kwargs["output_dir"]
            if "beta" in kwargs:
                if task == "dpo":
                    training_params["dpo_beta"] = kwargs["beta"]
                elif task == "kto":
                    training_params["kto_beta"] = kwargs["beta"]
            if "max_steps" in kwargs:
                warnings.append(
                    f"max_steps={kwargs['max_steps']}. "
                    "Soup uses epochs; set training.epochs instead."
                )

    if base is None:
        raise ValueError("No FastLanguageModel.from_pretrained() call found in notebook")

    if task == "cpo":
        # TRL's CPOTrainer is SimPO only under loss_type="simpo"; its other losses
        # (sigmoid, hinge, ipo over CPO's reference-free objective) have no Soup task.
        if cpo_loss_type is _UNREAD:
            raise ValueError(
                "No Soup task matches CPOTrainer: its loss_type could not be read statically "
                "(args= is not a CPOConfig(...) call at module level, or uses **kwargs); "
                "only loss_type='simpo' migrates, to task: simpo."
            )
        if cpo_loss_type != "simpo":
            raise ValueError(
                f"No Soup task matches CPOTrainer with loss_type={cpo_loss_type!r} "
                "(TRL's default is 'sigmoid'); only loss_type='simpo' migrates, to task: simpo."
            )
        task = "simpo"
    if task == "online_dpo":
        # The schema requires a judge or a reward model, and TRL's judge is a Python
        # object with no string form, so a placeholder is written the way data.train is.
        training_params.setdefault("online_dpo_judge", _ONLINE_DPO_JUDGE_PLACEHOLDER)
        warnings.append(
            "OnlineDPOTrainer's judge / reward model is not carried over: "
            f"training.online_dpo_judge is a placeholder ({_ONLINE_DPO_JUDGE_PLACEHOLDER}); "
            "set it to your judge, or replace it with training.reward_model."
        )

    # Build result
    data_format = _TASK_FORMAT_MAP.get(task, "auto")

    # Resolve quantization & full fine-tuning
    if full_finetuning:
        quantization = "none"
    elif load_in_16bit:
        quantization = "none"
    elif load_in_8bit:
        quantization = "8bit"
    elif load_in_4bit is False:
        quantization = "none"
    else:
        quantization = "4bit"

    training: Dict[str, Any] = {**training_params}
    training["quantization"] = quantization

    if full_finetuning:
        if task in ("sft", "embedding"):
            warnings.append(
                f"full_finetuning=True in Unsloth — no LoRA will be used "
                f"for task '{task}' (lora.r: 0)."
            )
            training["lora"] = {"r": 0}
        else:
            raise ValueError(
                f"full_finetuning is requested, but task '{task}' does not support full "
                "fine-tuning in Soup. Only sft and embedding support full fine-tuning (lora.r=0)."
            )
    elif lora_params:
        training["lora"] = lora_params

    data: Dict[str, Any] = {
        "train": "./data/train.jsonl",
        "format": data_format,
    }
    if max_seq_length:
        data["max_length"] = max_seq_length

    warnings.append("data.train is a placeholder — set the actual dataset path.")

    result: Dict[str, Any] = {
        "base": base,
        "task": task,
        "data": data,
        "training": training,
        "output": output_dir,
        "_warnings": warnings,
    }

    return result


class _UnreadType:
    """Marker: the trainer takes an ``args=`` that static reading cannot resolve."""


_UNREAD = _UnreadType()
_ONLINE_DPO_JUDGE_PLACEHOLDER = "ollama://REPLACE-ME"


def _cpo_loss_type(
    trainer_call: ast.Call,
    trainer_kwargs: Dict[str, Any],
    scope: Dict[str, Any],
    name_bindings: Dict[str, List[ast.AST]],
) -> Any:
    """The ``loss_type`` that governs *this* ``CPOTrainer``: on the call itself, else on
    the ``CPOConfig`` passed as its config, ``args=`` or the second positional argument
    (TRL's ``CPOTrainer(model, args, ...)``), inline or through a name whose latest
    module-level binding precedes the trainer call. A ``CPOConfig`` elsewhere in the
    notebook, or one bound to the name only after the trainer ran, does not count.
    ``None`` means omitted (TRL's default, ``sigmoid``); ``_UNREAD`` means a config was
    passed that cannot be read statically."""
    if "loss_type" in trainer_kwargs:
        return trainer_kwargs["loss_type"]
    config: Optional[ast.AST] = next(
        (kw.value for kw in trainer_call.keywords if kw.arg == "args"), None
    )
    if config is None and len(trainer_call.args) >= 2:
        config = trainer_call.args[1]
    if config is None:
        return None
    if isinstance(config, ast.Name):
        preceding = [
            value for value in name_bindings.get(config.id, [])
            if value.lineno < trainer_call.lineno
        ]
        config = preceding[-1] if preceding else None
    if not (isinstance(config, ast.Call) and _get_func_name(config) == "CPOConfig"):
        return _UNREAD
    if any(k.arg is None for k in config.keywords):  # CPOConfig(**kw)
        return _UNREAD
    return _extract_kwargs(config, scope=scope).get("loss_type")


def _get_func_name(node: ast.Call) -> Optional[str]:
    """Extract the function name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _extract_kwargs(
    node: ast.Call,
    scope: Optional[Dict[str, Any]] = None,
    unresolved_names: Optional[List[tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Extract keyword arguments from a function Call node as Python values.

    Extracts simple literal values (str, int, float, bool, list, None) or
    resolves variable names present in scope.
    """
    result: Dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue  # **kwargs — skip
        value = _ast_to_value(kw.value)
        if value is not _SENTINEL:
            result[kw.arg] = value
        elif isinstance(kw.value, ast.Name):
            if scope and kw.value.id in scope:
                result[kw.arg] = scope[kw.value.id]
            else:
                if unresolved_names is not None:
                    unresolved_names.append((kw.arg, kw.value.id))
    return result


class _SentinelType:
    """Sentinel for values that cannot be extracted."""
    pass


_SENTINEL = _SentinelType()


def _ast_to_value(node: ast.AST) -> Any:
    """Convert an AST node to a Python value. Returns _SENTINEL if not a literal."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.List):
        items = [_ast_to_value(elt) for elt in node.elts]
        if any(isinstance(item, _SentinelType) for item in items):
            return _SENTINEL
        return items
    elif isinstance(node, ast.Tuple):
        items = [_ast_to_value(elt) for elt in node.elts]
        if any(isinstance(item, _SentinelType) for item in items):
            return _SENTINEL
        return items
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _ast_to_value(node.operand)
        if isinstance(val, _SentinelType):
            return _SENTINEL
        return -val
    elif isinstance(node, ast.NameConstant):  # Python 3.7 compat
        return node.value
    return _SENTINEL
