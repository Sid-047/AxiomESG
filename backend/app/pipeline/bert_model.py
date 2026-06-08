from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _is_model_dir_complete(d: Path) -> bool:
    """Check that a model directory has config + weights + tokenizer."""
    if not d.exists():
        return False
    has_config = (d / "config.json").exists()
    has_weights = (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()
    has_tokenizer = (d / "tokenizer.json").exists() or (d / "vocab.txt").exists()
    return has_config and has_weights and has_tokenizer


def _get_model_dir() -> Optional[Path]:
    """
    Resolve the on-disk BERT ESG model directory relative to this backend.

    Prefers the v2 model only if it is complete, otherwise falls back to v1.
    Returns None if neither directory is complete.
    """
    backend_root = Path(__file__).resolve().parents[2]
    bert_root = backend_root / "bert_esg_classifier" / "content"

    v2_dir = bert_root / "bert_esg_classifier_v2"
    v1_dir = bert_root / "bert_esg_classifier"

    if _is_model_dir_complete(v2_dir):
        return v2_dir
    if _is_model_dir_complete(v1_dir):
        return v1_dir
    return None


# Lazy-loaded singleton
_tokenizer = None
_model = None
_loaded = False


def _ensure_loaded():
    global _tokenizer, _model, _loaded
    if _loaded:
        return
    model_dir = _get_model_dir()
    if model_dir is None:
        raise RuntimeError(
            "BERT ESG model not found or incomplete. "
            "Requires config.json + model weights + tokenizer files."
        )
    _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    _model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    _model.eval()
    _loaded = True


@torch.no_grad()
def predict(sentence: str) -> torch.Tensor:
    """
    Predict ESG class probabilities for a single sentence using the
    packaged BERT ESG classifier.

    Returns:
        probs: Tensor of shape [1, num_classes] with softmax probabilities.
    """
    _ensure_loaded()
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
    _ensure_loaded()
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

