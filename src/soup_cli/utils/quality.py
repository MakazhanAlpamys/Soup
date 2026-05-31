"""Data quality scoring — perplexity and coherence filters."""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


def compute_perplexity_scores(
    texts: list[str],
    model_name: str = "gpt2",
    batch_size: int = 8,
    max_length: int = 512,
) -> list[float]:
    """Compute perplexity scores for a list of texts using a language model.

    Lower perplexity = more "normal" text (the model finds it predictable).
    Very high perplexity may indicate garbage, corrupted, or very unusual text.

    Args:
        texts: List of text strings to score.
        model_name: HuggingFace model to use for perplexity (default: gpt2).
        batch_size: Batch size for inference.
        max_length: Max tokens per text.

    Returns:
        List of perplexity scores (one per text).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    scores = []

    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            # Per-token loss: outputs.loss is mean over all tokens
            # We need per-sample loss for individual perplexity scores
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            # Set pad positions to -100 so CrossEntropyLoss skips them
            shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())

            # Mask padding tokens and compute per-sample mean loss
            masked_loss = per_token_loss * shift_mask.float()
            sample_lengths = shift_mask.sum(dim=1).float().clamp(min=1)
            sample_loss = masked_loss.sum(dim=1) / sample_lengths

            for loss_val in sample_loss:
                ppl = math.exp(min(loss_val.item(), 100))  # cap to avoid overflow
                scores.append(ppl)

    # Cleanup GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores


def compute_coherence_score(texts: list[str]) -> list[float]:
    """Compute coherence scores for a list of texts.

    Coherence is measured by:
    - Sentence-level structure (has proper sentences)
    - Word repetition ratio (low repetition = more coherent)
    - Character diversity (not just repeated chars)

    Returns scores in [0, 1] where 1 = highly coherent.
    """
    scores = []
    for text in texts:
        score = _score_coherence(text)
        scores.append(score)
    return scores


def _score_coherence(text: str) -> float:
    """Score a single text for coherence. Returns 0.0–1.0."""
    if not text or not text.strip():
        return 0.0

    text = text.strip()
    words = text.split()

    if len(words) < 3:
        return 0.2

    # 1. Word repetition ratio (0-1, lower repetition = higher score)
    unique_words = set(w.lower() for w in words)
    repetition_score = len(unique_words) / len(words)  # 1.0 = all unique

    # 2. Sentence structure: check for sentence-ending punctuation
    sentence_endings = sum(1 for ch in text if ch in ".!?")
    sentence_score = min(1.0, sentence_endings / max(1, len(words) / 15))

    # 3. Character diversity: ratio of unique chars to total
    unique_chars = set(text.lower())
    char_diversity = min(1.0, len(unique_chars) / 30)  # 30+ unique chars = 1.0

    # 4. Average word length (very short or very long = less coherent)
    avg_word_len = sum(len(w) for w in words) / len(words)
    word_len_score = 1.0 if 3 <= avg_word_len <= 12 else 0.5

    # Weighted combination
    coherence = (
        0.35 * repetition_score
        + 0.25 * sentence_score
        + 0.20 * char_diversity
        + 0.20 * word_len_score
    )
    return round(min(1.0, max(0.0, coherence)), 4)


def filter_by_quality(
    data: list[dict],
    perplexity_threshold: Optional[float] = None,
    coherence_threshold: Optional[float] = None,
    perplexity_model: str = "gpt2",
    text_field: Optional[str] = None,
) -> tuple[list[dict], list[dict]]:
    """Filter dataset rows by quality scores.

    Args:
        data: List of data rows (dicts).
        perplexity_threshold: Max perplexity allowed (rows above this are removed).
        coherence_threshold: Min coherence required (rows below this are removed).
        perplexity_model: Model for perplexity scoring.
        text_field: Which field to score. None = concatenate all string values.

    Returns:
        (kept, removed) tuple of data lists.
    """
    if not data:
        return [], []

    # Extract texts
    texts = []
    for row in data:
        if text_field and text_field in row:
            texts.append(str(row[text_field]))
        else:
            texts.append(" ".join(str(v) for v in row.values() if v))

    # Compute scores
    perplexity_scores = None
    if perplexity_threshold is not None:
        perplexity_scores = compute_perplexity_scores(
            texts, model_name=perplexity_model,
        )

    coherence_scores = None
    if coherence_threshold is not None:
        coherence_scores = compute_coherence_score(texts)

    # Filter
    kept = []
    removed = []
    for idx, row in enumerate(data):
        remove = False
        if perplexity_scores is not None and perplexity_scores[idx] > perplexity_threshold:
            remove = True
        if coherence_scores is not None and coherence_scores[idx] < coherence_threshold:
            remove = True

        if remove:
            removed.append(row)
        else:
            kept.append(row)

    return kept, removed
