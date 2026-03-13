from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _get_model_dir() -> Path:
    """
    Resolve the on-disk BERT ESG model directory relative to this backend.

    Prefers the v2 model if present, otherwise falls back to v1.
    """
    backend_root = Path(__file__).resolve().parents[2]
    bert_root = backend_root / "bert_esg_classifier" / "content"

    v2_dir = bert_root / "bert_esg_classifier_v2"
    v1_dir = bert_root / "bert_esg_classifier"

    if v2_dir.exists():
        return v2_dir
    return v1_dir


_MODEL_DIR = _get_model_dir()

_tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
_model = AutoModelForSequenceClassification.from_pretrained(str(_MODEL_DIR))
_model.eval()


@torch.no_grad()
def predict(sentence: str) -> torch.Tensor:
    """
    Predict ESG class probabilities for a single sentence using the
    packaged BERT ESG classifier.

    Returns:
        probs: Tensor of shape [1, num_classes] with softmax probabilities.
    """
    inputs = _tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    outputs = _model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs


@torch.no_grad()
def batch_predict(sentences: List[str]) -> torch.Tensor:
    """
    Batched variant of `predict` for efficiency when scoring many sentences.

    Args:
        sentences: List of input sentences.

    Returns:
        probs: Tensor of shape [batch, num_classes].
    """
    if not sentences:
        return torch.empty(0, 3, dtype=torch.float32)

    inputs = _tokenizer(
        sentences,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    outputs = _model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs
