"""
Bridge between the BERT ESG classifier / fusion models and the main pipeline.

Provides four weighting strategies that use BERT embeddings + fusion:
  - bert_mean_weight     (MeanFusion)
  - bert_static_weight   (StaticFusion)
  - bert_awfa_v1_weight  (AWFAv1)
  - bert_awfa_v2_weight  (AWFAv2)

The BERT model is loaded lazily on first call to avoid penalizing
heuristic-only pipeline runs.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Lazy BERT model singleton
# ---------------------------------------------------------------------------

_bert_tokenizer = None
_bert_model = None
_bert_loaded = False


def _ensure_bert():
    """Load the BERT ESG classifier on first use."""
    global _bert_tokenizer, _bert_model, _bert_loaded
    if _bert_loaded:
        return
    from pathlib import Path
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    backend_root = Path(__file__).resolve().parents[2]
    bert_root = backend_root / "bert_esg_classifier" / "content"
    v2_dir = bert_root / "bert_esg_classifier_v2"
    v1_dir = bert_root / "bert_esg_classifier"

    model_dir = v2_dir if v2_dir.exists() else v1_dir
    if not model_dir.exists():
        raise RuntimeError(
            f"BERT ESG model not found at {model_dir}. "
            "BERT-based strategies require the model weights under "
            "backend/app/bert_esg_classifier/content/."
        )
    _bert_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    _bert_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    _bert_model.eval()
    _bert_loaded = True


@torch.no_grad()
def _bert_process(sentences: List[str]) -> Tuple[Tensor, Tensor]:
    """
    Run BERT over sentences and return CLS embeddings + ESG probabilities.

    Returns:
        embeddings:    [N, 768]
        probabilities: [N, 3]
    """
    _ensure_bert()
    if not sentences:
        return (
            torch.empty(0, 768, dtype=torch.float32),
            torch.empty(0, 3, dtype=torch.float32),
        )
    inputs = _bert_tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    outputs = _bert_model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    cls_embeddings = last_hidden[:, 0, :]  # [N, 768]
    probabilities = torch.softmax(outputs.logits, dim=-1)  # [N, 3]
    return cls_embeddings, probabilities


def _build_signals(embeddings: Tensor, probabilities: Tensor) -> List[Tensor]:
    """Build the three signals expected by fusion models."""
    if embeddings.numel() == 0:
        empty = torch.empty(0, 768, dtype=torch.float32)
        return [empty, empty.clone(), empty.clone()]

    num_sentences, feature_dim = embeddings.shape
    embedding_tensor = embeddings
    probability_tensor = probabilities.repeat(1, 256)  # [N, 768]
    confidence = probabilities.max(dim=-1).values  # [N]
    confidence_tensor = confidence.unsqueeze(-1).repeat(1, feature_dim)  # [N, 768]
    return [embedding_tensor, probability_tensor, confidence_tensor]


# ---------------------------------------------------------------------------
# Category label mapping: model index -> E/S/G key
# ---------------------------------------------------------------------------

_IDX_TO_CATEGORY = {0: "E", 1: "S", 2: "G"}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _flatten_sentences(
    category_sentences: Dict[str, List[str]],
) -> Tuple[List[str], List[str]]:
    """Flatten category dict into ordered (sentences, categories) lists."""
    sentences: List[str] = []
    categories: List[str] = []
    for category in ("E", "S", "G"):
        for s in category_sentences.get(category, []):
            sentences.append(s)
            categories.append(category)
    return sentences, categories


def _weights_to_tuples(
    sentences: List[str],
    categories: List[str],
    weights: Tensor,
) -> List[Tuple[str, str, float]]:
    """Convert per-sentence weight tensor to sorted, deduplicated result tuples."""
    if weights.numel() == 0:
        return []
    # Normalize weights to [0, 1]
    w_min = weights.min()
    w_max = weights.max()
    if w_max > w_min:
        norm_weights = (weights - w_min) / (w_max - w_min)
    else:
        norm_weights = torch.full_like(weights, 0.5)

    seen = set()
    result: List[Tuple[str, str, float]] = []
    for i, (sent, cat) in enumerate(zip(sentences, categories)):
        key = _normalize(sent)
        if not key or key in seen:
            continue
        seen.add(key)
        w = round(float(norm_weights[i].item()), 3)
        result.append((cat, sent, w))
    result.sort(key=lambda x: (-x[2], x[1]))
    return result


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def bert_mean_weight(
    category_sentences: Dict[str, List[str]],
) -> List[Tuple[str, str, float]]:
    """BERT + MeanFusion strategy."""
    from app.bert_esg_classifier.project.fusion.mean_fusion import MeanFusion

    sentences, categories = _flatten_sentences(category_sentences)
    if not sentences:
        return []
    embeddings, probabilities = _bert_process(sentences)
    fused, weights = MeanFusion()([embeddings, probabilities])
    return _weights_to_tuples(sentences, categories, weights)


def bert_static_weight(
    category_sentences: Dict[str, List[str]],
) -> List[Tuple[str, str, float]]:
    """BERT + StaticFusion strategy."""
    from app.bert_esg_classifier.project.fusion.static_fusion import StaticFusion

    sentences, categories = _flatten_sentences(category_sentences)
    if not sentences:
        return []
    embeddings, probabilities = _bert_process(sentences)
    fused, weights = StaticFusion()([embeddings, probabilities])
    return _weights_to_tuples(sentences, categories, weights)


def bert_awfa_v1_weight(
    category_sentences: Dict[str, List[str]],
) -> List[Tuple[str, str, float]]:
    """BERT + AWFAv1 strategy."""
    from app.bert_esg_classifier.project.fusion.awfa_v1 import AWFAv1

    sentences, categories = _flatten_sentences(category_sentences)
    if not sentences:
        return []
    embeddings, probabilities = _bert_process(sentences)
    signals = _build_signals(embeddings, probabilities)
    num_signals = len(signals)
    feature_dim = signals[0].shape[-1]
    model = AWFAv1(num_signals=num_signals, feature_dim=feature_dim)
    fused, signal_weights = model(signals)
    # AWFAv1 returns signal-level weights, not sentence-level.
    # Derive sentence importance from the fused representation norm.
    sentence_scores = torch.norm(fused, dim=-1) if fused.ndim == 2 else fused.abs()
    # If fused is a single vector (document-level), use per-sentence embedding norms
    if sentence_scores.numel() != len(sentences):
        sentence_scores = torch.norm(embeddings, dim=-1)
    return _weights_to_tuples(sentences, categories, sentence_scores)


def bert_awfa_v2_weight(
    category_sentences: Dict[str, List[str]],
) -> List[Tuple[str, str, float]]:
    """BERT + AWFAv2 strategy."""
    from app.bert_esg_classifier.project.fusion.awfa_v2 import AWFAv2

    sentences, categories = _flatten_sentences(category_sentences)
    if not sentences:
        return []
    embeddings, probabilities = _bert_process(sentences)
    signals = _build_signals(embeddings, probabilities)
    num_signals = len(signals)
    feature_dim = signals[0].shape[-1]
    model = AWFAv2(num_signals=num_signals, feature_dim=feature_dim)
    fused, signal_weights = model(signals)
    # Same approach as v1: derive sentence scores from fused representations
    sentence_scores = torch.norm(fused, dim=-1) if fused.ndim == 2 else fused.abs()
    if sentence_scores.numel() != len(sentences):
        sentence_scores = torch.norm(embeddings, dim=-1)
    return _weights_to_tuples(sentences, categories, sentence_scores)


# ---------------------------------------------------------------------------
# Register all BERT strategies in the strategy registry
# ---------------------------------------------------------------------------

from app.pipeline.strategies import register  # noqa: E402

register("bert_mean")(bert_mean_weight)
register("bert_static")(bert_static_weight)
register("bert_awfa_v1")(bert_awfa_v1_weight)
register("bert_awfa_v2")(bert_awfa_v2_weight)
