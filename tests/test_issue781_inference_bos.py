"""#781 — inference prompted tuned models with special tokens training never saw.

``soup chat``, ``soup serve`` (transformers backend), ``soup infer``,
``soup diff``, the MoLE serve runtime, the ``live_eval`` generators behind the
eval gates, and ``soup data generate``'s local provider all rendered the chat
template to a string and then tokenized that string with the tokenizer's default
``add_special_tokens=True``. A template that renders ``{{ bos_token }}``
(Llama-3, Gemma, Mistral) on a tokenizer whose post-processor also prepends BOS
therefore sent ``[bos, bos, ...]`` — reproduced on the real tokenizers in the
issue. Soup's default SFT path (``train_on_responses_only``) builds its ids with
``add_special_tokens=False`` (``data/loss_mask.py``), so a model tuned there was
prompted with a sequence it never trained on.

The invariant pinned here: when a template rendered the prompt, the tokenizer
adds nothing. That is what training does and what HF's own
``apply_chat_template(tokenize=True)`` does. It is deliberately NOT "drop a
duplicate BOS": Soup's ``data.chat_template`` presets render no BOS at all, so
their adapters trained with none and must be prompted with none.

Every tokenizer below is a real ``transformers`` fast tokenizer built offline
from an in-memory vocab, so ``apply_chat_template`` is the genuine Jinja
renderer and the BOS/EOS come from a genuine ``tokenizers`` post-processor — no
network, and nothing that could agree with a wrong encoding.
"""

import ast
from functools import lru_cache
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src" / "soup_cli"

_SPECIALS = [
    "<unk>", "<s>", "</s>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
    "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
]
_WORDS = [
    "You", "are", "terse", ".", "What", "is", "the", "capital", "of", "France", "?",
    "Paris", "And", "Italy", "Rome", "System", "User", "Assistant", ":",
    "system", "user", "assistant", "Generate", "1", "training", "examples", "now",
]
_BOS_ID = _SPECIALS.index("<s>")
_EOS_ID = _SPECIALS.index("</s>")

# Vendor shape (Llama-3 / Gemma / Mistral): the template renders BOS itself.
_BOS_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
# Same rendering with {% generation %} markers, so training takes loss_mask's
# preferred assistant-mask path rather than the incremental fallback.
_GENERATION_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}"
    "{% if m['role'] == 'assistant' %}"
    "<|assistant|> {% generation %}{{ m['content'] }} <|end|>{% endgeneration %} "
    "{% else %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
_REJECTS_SYSTEM_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}"
    "{% endif %}" + _BOS_TEMPLATE
)
_BROKEN_TEMPLATE = "{{ this_is_not_defined.boom() }}"

_SYSTEM = {"role": "system", "content": "You are terse."}
_USER = {"role": "user", "content": "What is the capital of France?"}
_ANSWER = {"role": "assistant", "content": "Paris."}
_USER_2 = {"role": "user", "content": "And Italy?"}
_ANSWER_2 = {"role": "assistant", "content": "Rome."}

_LEGACY = "System: You are terse.\nUser: What is the capital of France?\nAssistant:"


def _tokenizer(chat_template, *, post_processor="bos"):
    """A real fast tokenizer whose post-processor adds ``post_processor``.

    ``"bos"`` mirrors Llama-3 / Gemma / Mistral (``<s> $A``), ``"bos_eos"``
    a tokenizer that also appends EOS, ``None`` one that adds nothing.
    """
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers, processors

    vocab = {token: index for index, token in enumerate(_SPECIALS + _WORDS)}
    backend = tokenizers.Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.add_special_tokens(_SPECIALS)
    single = {"bos": "<s> $A", "bos_eos": "<s> $A </s>", None: None}[post_processor]
    if single is not None:
        backend.post_processor = processors.TemplateProcessing(
            single=single,
            special_tokens=[("<s>", _BOS_ID), ("</s>", _EOS_ID)],
        )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    tok.chat_template = chat_template
    return tok


def _preset_tokenizer(name):
    """A BOS-adding tokenizer carrying one of Soup's own ``data.chat_template``
    presets, applied through the same override ``soup train`` uses."""
    from soup_cli.data.chat_templates import apply_chat_template_override

    tok = _tokenizer(None)
    apply_chat_template_override(tok, name)
    return tok


def _hf_prompt_ids(tok, messages):
    """HF's own tokenize-in-one-step encoding — the reference."""
    return list(
        tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
    )


def _pre_fix_ids(tok, messages):
    """What every site sent before the fix: render to text, re-tokenize."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text)["input_ids"]


class _RecordingModel:
    """Stands in for the causal LM. Records the ids a call site hands to
    ``generate`` and echoes one extra token so the decode paths still run.

    Only the model is replaced; the tokenizer and the encoding are real."""

    def __init__(self):
        import torch

        self.device = torch.device("cpu")
        self.sent = None

    def eval(self):
        return self

    def generate(self, input_ids=None, **kwargs):
        import torch

        self.sent = input_ids[0].tolist()
        return torch.cat([input_ids, input_ids[:, -1:]], dim=1)


# ============================================================
# The shared encoder
# ============================================================


class TestEncodeChatPrompt:
    def test_the_fixture_reproduces_the_doubled_bos(self):
        """CONTROL for this whole file: the pre-fix encoding on this fixture
        really sends two BOS. Without it, a post-processor that silently added
        nothing would let every "exactly one BOS" assertion pass vacuously."""
        tok = _tokenizer(_BOS_TEMPLATE)

        assert _pre_fix_ids(tok, [_USER])[:2] == [_BOS_ID, _BOS_ID]

    def test_a_rendered_prompt_carries_exactly_one_bos(self):
        from soup_cli.utils.vllm import encode_chat_prompt

        tok = _tokenizer(_BOS_TEMPLATE)
        ids = encode_chat_prompt([_USER], tok, fallback_on_error=False)["input_ids"]

        assert ids[0] == _BOS_ID
        assert ids.count(_BOS_ID) == 1

    def test_ids_equal_hf_apply_chat_template_with_tokenize_true(self):
        from soup_cli.utils.vllm import encode_chat_prompt

        tok = _tokenizer(_BOS_TEMPLATE)
        messages = [_SYSTEM, _USER]

        ids = encode_chat_prompt(messages, tok, fallback_on_error=False)["input_ids"]

        assert ids == _hf_prompt_ids(tok, messages)

    def test_a_tokenizer_eos_is_not_appended_to_a_rendered_prompt(self):
        """A post-processor that appends EOS put an end-of-sequence token at the
        END of the prompt, i.e. right where generation starts."""
        from soup_cli.utils.vllm import encode_chat_prompt

        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos_eos")
        assert _pre_fix_ids(tok, [_USER])[-1] == _EOS_ID  # the defect, on this fixture

        ids = encode_chat_prompt([_USER], tok, fallback_on_error=False)["input_ids"]

        assert _EOS_ID not in ids
        assert ids == _hf_prompt_ids(tok, [_USER])

    def test_control_a_tokenizer_that_adds_nothing_is_byte_identical(self):
        """CONTROL — with no post-processor there was never a defect, so the
        fix must not move a single id."""
        from soup_cli.utils.vllm import encode_chat_prompt

        tok = _tokenizer(_BOS_TEMPLATE, post_processor=None)

        ids = encode_chat_prompt([_SYSTEM, _USER], tok, fallback_on_error=False)["input_ids"]

        assert ids == _pre_fix_ids(tok, [_SYSTEM, _USER])

    def test_a_soup_preset_on_a_bos_adding_tokenizer_is_prompted_with_no_bos(self):
        """Soup's presets render no BOS, and training adds none, so an adapter
        trained with ``data.chat_template: llama3`` never saw one. The pre-fix
        encoding still sent the tokenizer's.

        This is the case that separates the invariant from "drop BOS only when
        the rendered text already starts with it": that rule would leave this
        prompt with the BOS its training never had."""
        from soup_cli.utils.vllm import encode_chat_prompt

        tok = _preset_tokenizer("llama3")
        text = tok.apply_chat_template([_USER], tokenize=False, add_generation_prompt=True)
        assert not text.startswith("<s>")
        assert _pre_fix_ids(tok, [_USER])[0] == _BOS_ID  # the defect, on the preset

        ids = encode_chat_prompt([_USER], tok, fallback_on_error=False)["input_ids"]

        assert _BOS_ID not in ids
        assert ids == _hf_prompt_ids(tok, [_USER])

    def test_control_no_template_keeps_the_legacy_prompt_and_tokenizer_defaults(self):
        """CONTROL — a model that ships no template is served exactly as before:
        the legacy role-prefixed text, with whatever the tokenizer adds."""
        from soup_cli.utils.vllm import encode_chat_prompt

        tok = _tokenizer(None)

        ids = encode_chat_prompt([_SYSTEM, _USER], tok, fallback_on_error=False)["input_ids"]

        assert ids == tok(_LEGACY)["input_ids"]
        assert ids[0] == _BOS_ID

    def test_a_template_that_fails_to_render_falls_back_with_tokenizer_defaults(self):
        """The flag must mean "the template rendered this text", not "the
        tokenizer has a template". The fallback text never went through the
        template, so the tokenizer's own BOS is the only one it gets."""
        from soup_cli.utils.vllm import encode_chat_prompt

        tok = _tokenizer(_BROKEN_TEMPLATE)

        ids = encode_chat_prompt([_SYSTEM, _USER], tok, fallback_on_error=True)["input_ids"]

        assert ids == tok(_LEGACY)["input_ids"]
        assert ids[0] == _BOS_ID

    def test_a_template_that_fails_to_render_raises_when_fallback_is_off(self):
        jinja2 = pytest.importorskip("jinja2")
        from soup_cli.utils.vllm import encode_chat_prompt

        tok = _tokenizer(_BROKEN_TEMPLATE)

        with pytest.raises(jinja2.exceptions.TemplateError):
            encode_chat_prompt([_SYSTEM, _USER], tok, fallback_on_error=False)

    @pytest.mark.parametrize(
        ("template", "expected_kwargs"),
        [
            (_BOS_TEMPLATE, {"add_special_tokens": False, "return_tensors": "pt"}),
            # The legacy branch must be literally the call it always was: no
            # add_special_tokens key at all, since tokenizer stand-ins elsewhere
            # in the suite (and third-party tokenizer classes) take a fixed
            # signature.
            (None, {"return_tensors": "pt"}),
        ],
        ids=["rendered", "legacy"],
    )
    def test_the_tokenizer_receives_exactly_these_keywords(self, template, expected_kwargs):
        from soup_cli.utils.vllm import encode_chat_prompt

        inner = _tokenizer(template)
        seen = []

        class _Spy:
            chat_template = inner.chat_template

            def apply_chat_template(self, *args, **kwargs):
                return inner.apply_chat_template(*args, **kwargs)

            def __call__(self, text, **kwargs):
                seen.append(kwargs)
                return inner(text, **kwargs)

        encode_chat_prompt([_SYSTEM, _USER], _Spy(), fallback_on_error=False, return_tensors="pt")

        assert seen == [expected_kwargs]


# ============================================================
# The contract the fix exists for: inference == training
# ============================================================

_CONVERSATIONS = {
    "single_turn": ([_USER], _ANSWER),
    "with_system": ([_SYSTEM, _USER], _ANSWER),
    "multi_turn": ([_USER, _ANSWER, _USER_2], _ANSWER_2),
}


def _templated(kind):
    if kind == "vendor_bos":
        return _tokenizer(_BOS_TEMPLATE)
    if kind == "generation_markers":
        return _tokenizer(_GENERATION_TEMPLATE)
    return _preset_tokenizer("llama3")


class TestTrainInferAgreement:
    def test_the_fixtures_cover_both_training_mask_paths(self):
        """``loss_mask`` builds ids two ways; the agreement below means little
        if both fixtures took the same one."""
        from soup_cli.data.loss_mask import _apply_template_with_mask

        conversation = [_USER, _ANSWER]

        assert _apply_template_with_mask(_templated("generation_markers"), conversation)
        assert _apply_template_with_mask(_templated("vendor_bos"), conversation) is None

    @pytest.mark.parametrize("conversation", sorted(_CONVERSATIONS))
    @pytest.mark.parametrize("kind", ["generation_markers", "soup_llama3_preset", "vendor_bos"])
    def test_prompt_ids_are_the_prefix_soup_trained_on(self, kind, conversation):
        from soup_cli.data.loss_mask import build_assistant_only_labels
        from soup_cli.utils.vllm import encode_chat_prompt

        tok = _templated(kind)
        prompt, answer = _CONVERSATIONS[conversation]
        trained = build_assistant_only_labels(prompt + [answer], tok, max_length=512)

        ids = encode_chat_prompt(prompt, tok, fallback_on_error=False)["input_ids"]

        assert trained["input_ids"][: len(ids)] == ids
        assert len(trained["input_ids"]) > len(ids)

    @pytest.mark.parametrize("kind", ["soup_llama3_preset", "vendor_bos"])
    def test_control_the_pre_fix_encoding_was_not_that_prefix(self, kind):
        """CONTROL — the agreement above is only a finding if the old encoding
        disagreed on the same fixture."""
        from soup_cli.data.loss_mask import build_assistant_only_labels

        tok = _templated(kind)
        trained = build_assistant_only_labels([_USER, _ANSWER], tok, max_length=512)
        old = _pre_fix_ids(tok, [_USER])

        assert trained["input_ids"][: len(old)] != old


# ============================================================
# Every in-process generation path
# ============================================================


def _via_chat(tok, messages):
    from soup_cli.commands.chat import _generate

    model = _RecordingModel()
    _generate(model, tok, list(messages), max_tokens=1, temperature=0.0, device="cpu")
    return model.sent


def _via_serve(tok, messages):
    from soup_cli.commands.serve import _generate_response

    model = _RecordingModel()
    _generate_response(model, tok, list(messages), max_tokens=1, temperature=0.0)
    return model.sent


def _via_infer(tok, messages):
    from soup_cli.commands.infer import _generate

    model = _RecordingModel()
    _generate(model, tok, list(messages), max_tokens=1, temperature=0.0)
    return model.sent


def _via_diff(tok, messages):
    from soup_cli.commands.diff import _generate

    model = _RecordingModel()
    _generate(model, tok, list(messages), max_tokens=1, temperature=0.0)
    return model.sent


def _via_mole(tok, messages):
    """``LoadedMole.generate_text`` drives its own blended decode loop, so the
    recorder replaces that loop; the encoding in front of it is the real one."""
    from soup_cli.utils.mole_routing import LoadedMole

    loaded = LoadedMole(object(), tok, None, ["task_0", "task_1"])
    recorder = _RecordingModel()
    loaded.generate = lambda input_ids, attention_mask, **kwargs: recorder.generate(input_ids)
    loaded.generate_text(list(messages), max_tokens=1, temperature=0.0)
    return recorder.sent


_SITES = {
    "chat": _via_chat,
    "diff": _via_diff,
    "infer": _via_infer,
    "mole": _via_mole,
    "serve": _via_serve,
}


class TestEveryGenerationPathEncodesLikeTraining:
    @pytest.mark.parametrize("site", sorted(_SITES))
    def test_the_site_sends_hf_prompt_ids_with_one_bos(self, site):
        tok = _tokenizer(_BOS_TEMPLATE)
        messages = [_SYSTEM, _USER]

        sent = _SITES[site](tok, messages)

        assert sent == _hf_prompt_ids(tok, messages)
        assert sent.count(_BOS_ID) == 1

    @pytest.mark.parametrize("site", sorted(_SITES))
    def test_control_without_a_template_the_site_sends_the_legacy_prompt_as_before(self, site):
        tok = _tokenizer(None)

        assert _SITES[site](tok, [_SYSTEM, _USER]) == tok(_LEGACY)["input_ids"]

    @pytest.mark.parametrize("site", ["chat", "diff", "infer", "mole"])
    def test_a_template_that_rejects_the_conversation_still_raises(self, site):
        """These sites always let a template error surface (e.g. ``soup chat
        --system`` on a template with no system role). Routing them through the
        shared encoder must not turn that into a quietly served legacy prompt
        the model has never seen."""
        jinja2 = pytest.importorskip("jinja2")
        tok = _tokenizer(_REJECTS_SYSTEM_TEMPLATE)

        with pytest.raises(jinja2.exceptions.TemplateError):
            _SITES[site](tok, [_SYSTEM, _USER])

    def test_serve_still_falls_back_and_counts_the_tokens_it_actually_sent(self):
        from soup_cli.commands.serve import _generate_response

        tok = _tokenizer(_REJECTS_SYSTEM_TEMPLATE)
        model = _RecordingModel()

        _, prompt_tokens, _ = _generate_response(
            model, tok, [_SYSTEM, _USER], max_tokens=1, temperature=0.0
        )

        assert model.sent == tok(_LEGACY)["input_ids"]
        assert prompt_tokens == len(model.sent)

    def test_serve_usage_prompt_tokens_matches_the_rendered_prompt(self):
        """``usage.prompt_tokens`` is read off the encoded ids, so it drops by
        the duplicated BOS — pinned so the reported number is the sent one."""
        from soup_cli.commands.serve import _generate_response

        tok = _tokenizer(_BOS_TEMPLATE)
        model = _RecordingModel()

        _, prompt_tokens, _ = _generate_response(
            model, tok, [_SYSTEM, _USER], max_tokens=1, temperature=0.0
        )

        assert prompt_tokens == len(_hf_prompt_ids(tok, [_SYSTEM, _USER]))


class TestLiveEvalGenerators:
    """The closures behind training-time eval gates, ``soup ship``,
    ``soup diagnose`` and ``soup advise``."""

    @staticmethod
    def _sent(factory_name, tok, prompt):
        from soup_cli.utils import live_eval

        model = _RecordingModel()
        factory = getattr(live_eval, factory_name)
        generate = factory("unused", loaded=(model, tok, "cpu"))
        if factory_name == "make_multi_generator":
            generate(prompt, 1)
        else:
            generate(prompt)
        return model.sent

    @pytest.mark.parametrize("factory_name", ["make_generator", "make_multi_generator"])
    def test_the_generator_sends_hf_prompt_ids_with_one_bos(self, factory_name):
        tok = _tokenizer(_BOS_TEMPLATE)

        sent = self._sent(factory_name, tok, _USER["content"])

        assert sent == _hf_prompt_ids(tok, [_USER])
        assert sent.count(_BOS_ID) == 1

    @pytest.mark.parametrize("factory_name", ["make_generator", "make_multi_generator"])
    @pytest.mark.parametrize("template", [None, _BROKEN_TEMPLATE], ids=["no-template", "broken"])
    def test_control_an_unrendered_raw_prompt_is_tokenized_as_before(
        self, factory_name, template
    ):
        """CONTROL — with no template, or one that fails to render, these
        generators send the raw prompt; no template touched it, so the
        tokenizer's own special tokens stay."""
        tok = _tokenizer(template)

        sent = self._sent(factory_name, tok, _USER["content"])

        assert sent == tok(_USER["content"])["input_ids"]


class TestDataGenerateLocalProvider:
    _PROMPT = "You are terse."
    _REQUEST = "Generate 1 training examples now."

    def _sent(self, monkeypatch, tok):
        transformers = pytest.importorskip("transformers")
        from soup_cli.commands import generate

        model = _RecordingModel()
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
        monkeypatch.setattr(
            transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model
        )
        monkeypatch.setattr(
            "soup_cli.utils.trust_remote.model_requires_trust_remote_code", lambda *a, **k: False
        )
        monkeypatch.setattr(
            "soup_cli.utils.trust_remote.resolve_trust_remote_code", lambda *a, **k: False
        )
        generate._generate_local(
            prompt="unused",
            count=1,
            fmt="alpaca",
            model_name="local-model",
            temperature=0.0,
            seed_examples=[],
            generation_prompt=self._PROMPT,
        )
        return model.sent

    def test_the_local_provider_sends_hf_prompt_ids_with_one_bos(self, monkeypatch):
        tok = _tokenizer(_BOS_TEMPLATE)
        messages = [
            {"role": "system", "content": self._PROMPT},
            {"role": "user", "content": self._REQUEST},
        ]

        sent = self._sent(monkeypatch, tok)

        assert sent == _hf_prompt_ids(tok, messages)
        assert sent.count(_BOS_ID) == 1

    def test_control_without_a_template_its_own_prompt_is_tokenized_as_before(
        self, monkeypatch
    ):
        tok = _tokenizer(None)

        sent = self._sent(monkeypatch, tok)

        assert sent == tok(f"{self._PROMPT}\n\n{self._REQUEST}\n\n")["input_ids"]


# ============================================================
# Ratchet: no new scope may render a template and re-add special tokens
# ============================================================

# Functions whose return value is text a chat template rendered.
_RENDER_HELPERS = frozenset({
    "_apply_prompt_template",
    "_render_chat_prompt",
    "_render_prompt_template",
    "build_chat_prompt",
    "render_raft_prompt",
})
_TOKENIZER_NAMES = frozenset({"_tokenizer", "processor", "teacher_tokenizer", "tok", "tokenizer"})

# Scopes that render and then let the tokenizer decide, each for a stated reason.
_ALLOWLIST = {
    "trainer/sft.py::VisionLanguageDataCollator.__call__": (
        "passes add_special_tokens through **processor_kwargs, decided per processor "
        "by _processor_adds_leading_bos (#302)"
    ),
    # preprocess_dataset was fixed in #785: its chat branch now tokenizes with
    # add_special_tokens=is_pretrain (False for templated rows), so it no longer
    # renders-then-adds and has left the allowlist.
    "utils/live_eval.py::extract_layer_activations": (
        "activation capture; changing the ids changes saved steering vectors and "
        "probe calibration (#781, out of scope)"
    ),
    "utils/steering.py::_capture_attn_heads": (
        "activation capture, same reason as extract_layer_activations (#781)"
    ),
}


def _call_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _renders(call):
    name = _call_name(call.func)
    if name in _RENDER_HELPERS:
        return True
    return name == "apply_chat_template" and any(
        keyword.arg == "tokenize"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in call.keywords
    )


def _tokenizes(call):
    func = call.func
    if _call_name(func) in _TOKENIZER_NAMES:
        return True
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "encode"
        and _call_name(func.value) in _TOKENIZER_NAMES
    )


def _adds_special_tokens(call):
    for keyword in call.keywords:
        if keyword.arg == "add_special_tokens":
            return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
    return True  # the tokenizer default


_SCOPE_NODES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _scopes_of(source):
    """Per qualified function name (nested functions are their own scope): does it
    render a template, and which tokenizer calls does it make?

    Known blind spot, stated rather than hidden: a render in one function whose
    text is tokenized in a DIFFERENT function is not seen. Every site this issue
    found does both in one scope."""
    scopes = {}
    stack = [(ast.parse(source), "<module>")]
    while stack:
        node, name = stack.pop()
        scope = scopes.setdefault(name, {"renders": False, "adds": [], "exact": []})
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPE_NODES):
                qualname = child.name if name == "<module>" else f"{name}.{child.name}"
                stack.append((child, qualname))
                continue
            if isinstance(child, ast.Call):
                if _renders(child):
                    scope["renders"] = True
                if _tokenizes(child):
                    key = "adds" if _adds_special_tokens(child) else "exact"
                    scope[key].append(child.lineno)
            stack.append((child, name))
    return scopes


def _findings(scopes):
    return sorted(name for name, scope in scopes.items() if scope["renders"] and scope["adds"])


@lru_cache(maxsize=1)
def _tree_scopes():
    result = {}
    for path in sorted(_SRC.rglob("*.py")):
        module = path.relative_to(_SRC).as_posix()
        for name, scope in _scopes_of(path.read_text(encoding="utf-8")).items():
            result[f"{module}::{name}"] = scope
    return result


class TestNoRenderThenReTokenize:
    def test_no_scope_renders_a_template_and_lets_the_tokenizer_add_special_tokens(self):
        found = set(_findings(_tree_scopes()))

        assert found - set(_ALLOWLIST) == set()

    def test_every_allowlist_entry_is_still_earned(self):
        """An entry whose scope was fixed, renamed or moved must leave the
        allowlist, or it quietly becomes a hole."""
        found = set(_findings(_tree_scopes()))

        assert set(_ALLOWLIST) - found == set()

    def test_the_scanner_sees_a_scope_that_already_does_it_right(self):
        """Not vacuous: ``_tokenize_pair`` renders and tokenizes in one scope,
        so a scanner that could not see either would still report zero."""
        scope = _tree_scopes()["utils/live_eval.py::_tokenize_pair"]

        assert scope["renders"]
        assert scope["exact"]
        assert not scope["adds"]

    @pytest.mark.parametrize(
        "source",
        [
            # chat / infer / diff before the fix
            "def _generate(model, tokenizer, messages):\n"
            "    text = tokenizer.apply_chat_template(\n"
            "        messages, tokenize=False, add_generation_prompt=True\n"
            "    )\n"
            "    return tokenizer(text, return_tensors='pt')\n",
            # serve before the fix: the shared builder, then a bare call
            "def _generate_response(model, tokenizer, messages):\n"
            "    text = build_chat_prompt(messages, tokenizer)\n"
            "    return tokenizer(text, return_tensors='pt')\n",
            # live_eval before the fix: a nested closure
            "def make_generator(tokenizer):\n"
            "    def _gen(prompt):\n"
            "        text = _apply_prompt_template(tokenizer, prompt)\n"
            "        return tokenizer(text, return_tensors='pt', truncation=True)\n"
            "    return _gen\n",
            # the MoLE runtime shape: a method on self.tokenizer
            "class Runtime:\n"
            "    def generate_text(self, messages):\n"
            "        text = self.tokenizer.apply_chat_template(messages, tokenize=False)\n"
            "        return self.tokenizer(text, return_tensors='pt')\n",
            # an explicit True is the same defect spelled out
            "def f(tokenizer, messages):\n"
            "    text = tokenizer.apply_chat_template(messages, tokenize=False)\n"
            "    return tokenizer(text, add_special_tokens=True)\n",
            # .encode adds special tokens by default too
            "def f(tok, messages):\n"
            "    text = tok.apply_chat_template(messages, tokenize=False)\n"
            "    return tok.encode(text)\n",
        ],
        ids=["inline-render", "shared-builder", "nested-closure", "self-tokenizer",
             "explicit-true", "encode"],
    )
    def test_the_scanner_flags_each_pre_fix_shape(self, source):
        assert len(_findings(_scopes_of(source))) == 1

    @pytest.mark.parametrize(
        "source",
        [
            "def f(tokenizer, messages):\n"
            "    text = tokenizer.apply_chat_template(messages, tokenize=False)\n"
            "    return tokenizer(text, add_special_tokens=False)\n",
            "def f(tokenizer, messages):\n"
            "    return encode_chat_prompt(messages, tokenizer, fallback_on_error=False)\n",
            "def f(tokenizer, messages):\n"
            "    text, templated = _render_prompt_template(tokenizer, messages)\n"
            "    return encode_rendered_prompt(tokenizer, text, templated=templated)\n",
            # tokenizing text no template rendered is not this defect
            "def f(tokenizer, text):\n"
            "    return tokenizer(text, return_tensors='pt')\n",
            # tokenize=True is HF's one-step encoding, not a render
            "def f(tokenizer, messages, text):\n"
            "    ids = tokenizer.apply_chat_template(messages, tokenize=True)\n"
            "    return tokenizer(text)\n",
        ],
        ids=["explicit-false", "shared-encoder", "rendered-encoder", "no-render",
             "tokenize-true"],
    )
    def test_the_scanner_passes_the_fixed_shapes(self, source):
        assert _findings(_scopes_of(source)) == []

    def test_the_legacy_role_prefixed_prompt_exists_only_in_the_shared_fallback(self):
        """chat / infer / diff / mole each carried their own copy of the
        role-prefixed fallback. One copy left means one place to be right."""
        literal = 'f"User: {content}"'
        copies = {
            path.relative_to(_SRC).as_posix(): path.read_text(encoding="utf-8").count(literal)
            for path in sorted(_SRC.rglob("*.py"))
        }

        assert {module: count for module, count in copies.items() if count} == {
            "utils/vllm.py": 1
        }
