"""Catastrophic forgetting detection (Part G of v0.25.0).

Runs lightweight mini benchmarks against a model during training and flags
significant drops in general-knowledge accuracy from the pre-training baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal, Optional

MiniBenchmark = list[dict[str, str]]


# ---------------------------------------------------------------------------
# Answer scoring (v0.71.38 — "the gate grows teeth")
# ---------------------------------------------------------------------------
#
# The v0.25.0 scorer was ``answer.lower() in output.lower()`` — a raw substring
# test that credited "B" for "**B**erlin"/"**b**ecause", "ok" for "lo**ok**",
# and "3" for "1**3**". That made the whole leg-2 regression gate decorative.
# The fix is answer-EXTRACTION (pull the model's chosen option) + a
# boundary-aware match, so a spurious substring no longer scores.

# MCQ option letters run A–J (the bundled suites never exceed 5 options today,
# but the extractor is generous so richer suites drop in without a scorer edit).
_MCQ_OPTIONS = "ABCDEFGHIJ"

# Cap the model output the scorer regexes scan (defense-in-depth — the regexes
# are all O(n) with bounded quantifiers, but ``score_answer`` is public SDK
# surface callable with an arbitrary, unbounded ``generate_fn``).
_MAX_SCORE_OUTPUT = 64 * 1024

# Explicit answer cue: "answer is C", "correct option: B", "I pick (A)". The
# optional ``is`` lets the letter sit one filler word past the cue.
_CUE_RE = re.compile(
    r"(?:answer|correct|option|choice|select|pick)\b[\s:=.\-]{0,4}(?:is\s+)?"
    r"\(?([A-Ja-j])\)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# Parenthesised choice marker: "(B)" or "B)".
_PAREN_RE = re.compile(r"(?<![A-Za-z0-9])\(?([A-Ja-j])\)(?![A-Za-z0-9])")
# Bare standalone UPPERCASE letter that TERMINATES a clause — i.e. followed by
# end-of-string or sentence punctuation, not by more words. Requiring upper-case
# skips the lower-case article "a"/"an", and the terminating look-ahead skips a
# capitalized sentence-opener like "A cat sat ..." / the pronoun "I think ..."
# that a bare "\b" match would wrongly read as choosing option A / I.
_BARE_UPPER_RE = re.compile(r"(?<![A-Za-z0-9])([A-J])(?=[.,;:!?]|\s*$)")


def _is_mcq_letter(answer: str) -> bool:
    """True when ``answer`` is a single MCQ option letter (A–J)."""
    return len(answer) == 1 and answer.upper() in _MCQ_OPTIONS


def extract_mcq_letter(output: str) -> Optional[str]:
    """Pull the model's chosen MCQ option letter (A–J) from ``output``.

    Priority: an explicit ``answer/option/...`` cue, then a parenthesised
    ``(B)``/``B)`` marker, then a bare clause-terminating uppercase letter. The
    cue and paren tiers take the LAST match (a model echoes the option list
    before deciding, so its real choice is at the end); the bare tier takes the
    FIRST match (a model answering without a cue leads with the letter, e.g.
    "C. I think ..."). A bare letter only counts when it terminates a clause, so
    a prose opener ("A cat sat ...") is not read as choosing option A. Returns
    ``None`` when no option letter is present (e.g. a free-text answer like
    "Berlin").
    """
    if not isinstance(output, str) or not output:
        return None
    output = output[:_MAX_SCORE_OUTPUT]
    for regex in (_CUE_RE, _PAREN_RE):
        matches = regex.findall(output)
        if matches:
            return matches[-1].upper()
    bare = _BARE_UPPER_RE.findall(output)
    if bare:
        return bare[0].upper()
    return None


def _standalone_match(output: str, answer: str) -> bool:
    """True when ``answer`` appears in ``output`` as a whole alnum-bounded token.

    Uses alnum look-around instead of ``\\b`` so an answer that begins or ends
    with punctuation still matches cleanly; case-insensitive. This is what stops
    "ok" scoring for "lo**ok**" and "3" for "1**3**".
    """
    pattern = r"(?<![A-Za-z0-9])" + re.escape(answer) + r"(?![A-Za-z0-9])"
    return re.search(pattern, output[:_MAX_SCORE_OUTPUT], re.IGNORECASE) is not None


def score_answer(output: str, answer: str) -> bool:
    """Score one model ``output`` against a mini-benchmark ``answer``.

    MCQ letter answers are scored by extraction + exact letter match; every
    other answer by a boundary-aware standalone-token match. Replaces the
    v0.25.0 raw-substring scorer (the "gate grows teeth" fix, v0.71.38).
    """
    if not isinstance(output, str) or not isinstance(answer, str):
        return False
    ans = answer.strip()
    if not ans:
        return False
    if _is_mcq_letter(ans):
        chosen = extract_mcq_letter(output)
        return chosen is not None and chosen == ans.upper()
    return _standalone_match(output, ans)


# ---------------------------------------------------------------------------
# Built-in mini benchmarks (v0.71.38 — expanded from the v0.25.0 5-item
# starter sets so a single-item flip (1/N < 0.05) actually trips the ship gate
# rather than being rounded away).
#
# PROVENANCE / decontamination: every item below is ORIGINAL, hand-authored for
# Soup. NO row is copied from MMLU / GSM8K / HellaSwag / any public benchmark —
# the gate suite must itself pass ``soup data decontaminate``. Answers are
# scored by ``score_answer`` (extraction + boundary match), never raw substring.
# ---------------------------------------------------------------------------

MINI_MMLU: MiniBenchmark = [
    {"question": "What is 2 + 2? (A) 3 (B) 4 (C) 5", "answer": "B"},
    {"question": "The capital of France is: (A) London (B) Berlin (C) Paris", "answer": "C"},
    {"question": "Water freezes at: (A) 0C (B) 50C (C) 100C", "answer": "A"},
    {"question": "Photosynthesis mainly uses: (A) oxygen (B) carbon dioxide (C) nitrogen",
     "answer": "B"},
    {"question": "Atomic number of hydrogen is: (A) 1 (B) 2 (C) 3", "answer": "A"},
    {"question": "The largest planet in our solar system is: (A) Earth (B) Jupiter (C) Mars",
     "answer": "B"},
    {"question": "The chemical symbol for gold is: (A) Gd (B) Go (C) Au", "answer": "C"},
    {"question": "How many continents are there on Earth? (A) 5 (B) 7 (C) 9", "answer": "B"},
    {"question": "The powerhouse of the cell is the: (A) nucleus (B) ribosome (C) mitochondrion",
     "answer": "C"},
    {"question": "Light travels faster than: (A) sound (B) nothing (C) radio waves",
     "answer": "A"},
    {"question": "The square root of 81 is: (A) 8 (B) 9 (C) 18", "answer": "B"},
    {"question": "Which gas do plants release during photosynthesis? "
                 "(A) oxygen (B) hydrogen (C) methane", "answer": "A"},
    {"question": "The author of 'Romeo and Juliet' is: (A) Dickens (B) Shakespeare (C) Austen",
     "answer": "B"},
    {"question": "The freezing point of water in Fahrenheit is: (A) 0 (B) 32 (C) 100",
     "answer": "B"},
    {"question": "Which ocean is the largest? (A) Atlantic (B) Indian (C) Pacific",
     "answer": "C"},
    {"question": "The human body has how many pairs of ribs? (A) 10 (B) 12 (C) 14",
     "answer": "B"},
    {"question": "The currency of Japan is the: (A) yuan (B) won (C) yen", "answer": "C"},
    {"question": "Which metal is liquid at room temperature? (A) mercury (B) iron (C) copper",
     "answer": "A"},
    {"question": "The speed of light is closest to: (A) 300 km/s (B) 300,000 km/s (C) 30 km/s",
     "answer": "B"},
    {"question": "A triangle has how many sides? (A) 3 (B) 4 (C) 5", "answer": "A"},
    {"question": "The closest star to Earth is the: (A) Moon (B) Sun (C) Sirius", "answer": "B"},
    {"question": "DNA carries: (A) genetic information (B) oxygen (C) sugar", "answer": "A"},
    {"question": "The tallest mountain above sea level is: (A) K2 (B) Everest (C) Denali",
     "answer": "B"},
    {"question": "Which is a prime number? (A) 9 (B) 15 (C) 13", "answer": "C"},
    {"question": "The primary gas in Earth's atmosphere is: (A) oxygen (B) nitrogen (C) argon",
     "answer": "B"},
    {"question": "Sound cannot travel through a: (A) solid (B) liquid (C) vacuum", "answer": "C"},
]

MINI_COMMON_SENSE: MiniBenchmark = [
    {"question": "If it is raining, you should bring a: (A) hat (B) umbrella (C) fan",
     "answer": "B"},
    {"question": "You eat breakfast in the: (A) morning (B) evening (C) night", "answer": "A"},
    {"question": "Fish live in: (A) trees (B) water (C) sand", "answer": "B"},
    {"question": "The sun rises in the: (A) west (B) south (C) east", "answer": "C"},
    {"question": "Ice melts when: (A) heated (B) frozen (C) pressed", "answer": "A"},
    {"question": "To unlock a door you usually need a: (A) spoon (B) key (C) pillow",
     "answer": "B"},
    {"question": "Before crossing a busy street you should: (A) close your eyes "
                 "(B) look both ways (C) run fast", "answer": "B"},
    {"question": "If a glass falls on tile it will likely: (A) bounce (B) break (C) float",
     "answer": "B"},
    {"question": "To write on paper you use a: (A) hammer (B) pen (C) plate", "answer": "B"},
    {"question": "A wet towel is best dried by: (A) folding it (B) hanging it in the sun "
                 "(C) burying it", "answer": "B"},
    {"question": "You feel cold, so you put on a: (A) swimsuit (B) coat (C) sandal",
     "answer": "B"},
    {"question": "Plants need this to grow: (A) darkness (B) sunlight (C) television",
     "answer": "B"},
    {"question": "To call a friend far away you use a: (A) phone (B) fork (C) broom",
     "answer": "A"},
    {"question": "After you finish eating, dirty dishes should be: (A) washed (B) painted "
                 "(C) planted", "answer": "A"},
    {"question": "If the room is dark you should turn on the: (A) faucet (B) light (C) oven",
     "answer": "B"},
    {"question": "A baby is younger than a: (A) grandparent (B) newborn (C) egg", "answer": "A"},
    {"question": "To cut paper you use: (A) scissors (B) a magnet (C) glue", "answer": "A"},
    {"question": "Bread that is very old may become: (A) fresh (B) moldy (C) cold",
     "answer": "B"},
    {"question": "If you are thirsty you should drink: (A) sand (B) water (C) paper",
     "answer": "B"},
    {"question": "A car needs this to run: (A) fuel (B) music (C) paint", "answer": "A"},
    {"question": "You wear shoes on your: (A) hands (B) head (C) feet", "answer": "C"},
    {"question": "To keep food cold you put it in the: (A) oven (B) refrigerator (C) closet",
     "answer": "B"},
    {"question": "If you are tired at night you should: (A) sleep (B) exercise hard (C) shout",
     "answer": "A"},
    {"question": "Rain falls from the: (A) ground (B) clouds (C) ocean floor", "answer": "B"},
]

MINI_INSTRUCTION: MiniBenchmark = [
    {"question": "Respond with just the word: ok", "answer": "ok"},
    {"question": "Answer in one word — the color of grass:", "answer": "green"},
    {"question": "Answer yes or no: Is fire hot?", "answer": "yes"},
    {"question": "Reply with only the number three.", "answer": "3"},
    {"question": "Say only the word: done", "answer": "done"},
    {"question": "Reply with a single word: the opposite of 'up'.", "answer": "down"},
    {"question": "Answer with one word: the color of a clear daytime sky.", "answer": "blue"},
    {"question": "Answer yes or no: Is water wet?", "answer": "yes"},
    {"question": "Respond with only the word: apple", "answer": "apple"},
    {"question": "Reply with the number that comes after 9.", "answer": "10"},
    {"question": "Answer in one word: what animal says 'meow'?", "answer": "cat"},
    {"question": "Say only the word: banana", "answer": "banana"},
    {"question": "Answer with one word: the opposite of 'hot'.", "answer": "cold"},
    {"question": "Reply with the single word: yes if 2 is greater than 1.", "answer": "yes"},
    {"question": "Answer in one word: the first month of the year.", "answer": "January"},
    {"question": "Respond with only: purple", "answer": "purple"},
    {"question": "Answer with one word: what do bees make?", "answer": "honey"},
    {"question": "Reply with only the number of days in a week.", "answer": "7"},
    {"question": "Answer in one word: the opposite of 'left'.", "answer": "right"},
    {"question": "Say only the word: ocean", "answer": "ocean"},
    {"question": "Answer with one word: what color is a ripe tomato?", "answer": "red"},
    {"question": "Reply with a single word: the season after winter.", "answer": "spring"},
    {"question": "Answer in one word: what do you call frozen water?", "answer": "ice"},
    {"question": "Respond with only the number: how many wheels does a bicycle have?",
     "answer": "2"},
]

# Arithmetic — the market asks for basic numeracy; scored by extraction + exact.
MINI_ARITHMETIC: MiniBenchmark = [
    {"question": "What is 6 + 7?", "answer": "13"},
    {"question": "What is 9 - 4?", "answer": "5"},
    {"question": "What is 8 times 3?", "answer": "24"},
    {"question": "What is 6 multiplied by 7?", "answer": "42"},
    {"question": "What is 12 divided by 4?", "answer": "3"},
    {"question": "What is 15 + 27?", "answer": "42"},
    {"question": "What is 100 - 37?", "answer": "63"},
    {"question": "What is 9 times 9?", "answer": "81"},
    {"question": "What is half of 50?", "answer": "25"},
    {"question": "What is 7 + 8?", "answer": "15"},
    {"question": "What is 5 squared?", "answer": "25"},
    {"question": "What is 45 divided by 9?", "answer": "5"},
    {"question": "What is 11 times 11?", "answer": "121"},
    {"question": "What is 30 - 12?", "answer": "18"},
    {"question": "What is 4 + 4 + 4?", "answer": "12"},
    {"question": "What is 3 times 12?", "answer": "36"},
    {"question": "What is 200 - 99?", "answer": "101"},
    {"question": "What is 8 + 9?", "answer": "17"},
    {"question": "What is 72 divided by 8?", "answer": "9"},
    {"question": "What is 14 times 2?", "answer": "28"},
    {"question": "What is 6 + 6 + 6?", "answer": "18"},
    {"question": "What is 10 percent of 200?", "answer": "20"},
    {"question": "What is 16 + 25?", "answer": "41"},
    {"question": "What is 90 divided by 3?", "answer": "30"},
    {"question": "What is 7 times 6?", "answer": "42"},
    {"question": "What is 50 + 50?", "answer": "100"},
    {"question": "What is 13 - 7?", "answer": "6"},
    {"question": "What is 4 times 25?", "answer": "100"},
    {"question": "What is 18 + 24?", "answer": "42"},
    {"question": "What is 64 divided by 8?", "answer": "8"},
    {"question": "What is 9 times 8?", "answer": "72"},
    {"question": "What is 1000 - 1?", "answer": "999"},
    {"question": "What is 3 cubed?", "answer": "27"},
    {"question": "What is 40 + 60?", "answer": "100"},
    {"question": "What is 21 divided by 7?", "answer": "3"},
    {"question": "What is 12 times 12?", "answer": "144"},
]

MINI_BENCHMARKS: dict[str, MiniBenchmark] = {
    "mini_mmlu": MINI_MMLU,
    "mini_common_sense": MINI_COMMON_SENSE,
    "mini_instruction": MINI_INSTRUCTION,
    "mini_arithmetic": MINI_ARITHMETIC,
}


@dataclass
class ForgettingResult:
    """Outcome of a single forgetting eval."""

    step: int
    accuracy: float
    baseline: float
    delta: float
    warning_level: Literal["green", "yellow", "red"]


class ForgettingDetector:
    """Run a mini benchmark periodically and report accuracy drops."""

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        benchmark: str = "mini_mmlu",
        threshold: float = 0.10,
    ) -> None:
        if benchmark not in MINI_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark '{benchmark}'. "
                f"Options: {', '.join(MINI_BENCHMARKS.keys())}"
            )
        self.generate_fn = generate_fn
        self.benchmark_name = benchmark
        self.benchmark = MINI_BENCHMARKS[benchmark]
        self.threshold = threshold
        self._baseline_accuracy: Optional[float] = None

    def _evaluate(self) -> float:
        correct = 0
        for item in self.benchmark:
            output = self.generate_fn(item["question"])
            if not isinstance(output, str):
                continue
            if score_answer(output, item["answer"]):
                correct += 1
        return correct / len(self.benchmark) if self.benchmark else 0.0

    def run_baseline(self) -> float:
        """Compute the baseline accuracy before training starts."""
        self._baseline_accuracy = self._evaluate()
        return self._baseline_accuracy

    def _build_result(self, step: int, accuracy: float) -> ForgettingResult:
        baseline = self._baseline_accuracy if self._baseline_accuracy is not None else accuracy
        delta = baseline - accuracy
        if delta <= self.threshold:
            level: Literal["green", "yellow", "red"] = "green"
        elif delta <= self.threshold * 2:
            level = "yellow"
        else:
            level = "red"
        return ForgettingResult(
            step=step,
            accuracy=accuracy,
            baseline=baseline,
            delta=delta,
            warning_level=level,
        )

    def check_forgetting(self, step: int) -> ForgettingResult:
        """Run the mini benchmark and compare against the baseline."""
        if self._baseline_accuracy is None:
            self.run_baseline()
        current = self._evaluate()
        return self._build_result(step=step, accuracy=current)
