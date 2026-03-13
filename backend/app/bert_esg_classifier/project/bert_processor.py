from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Resolve model directory relative to this file so it works
# regardless of the working directory used to run the pipeline.
_THIS_DIR = Path(__file__).resolve().parent
_BERT_ROOT = _THIS_DIR.parent  # points to `bert_esg_classifier`

# Prefer the v2 model if available, otherwise fall back to v1.
_MODEL_DIR_V2 = _BERT_ROOT / "content" / "bert_esg_classifier_v2"
_MODEL_DIR_V1 = _BERT_ROOT / "content" / "bert_esg_classifier"

if _MODEL_DIR_V2.exists():
    _MODEL_DIR = _MODEL_DIR_V2
else:
    _MODEL_DIR = _MODEL_DIR_V1


_tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
_model = AutoModelForSequenceClassification.from_pretrained(str(_MODEL_DIR))
_model.eval()


@torch.no_grad()
def process_sentences(sentences: List[str]) -> Tuple[Tensor, Tensor]:
    """
    Run BERT over a batch of sentences to obtain:
      - CLS embeddings (shape: [num_sentences, 768])
      - ESG class probabilities (shape: [num_sentences, 3])

    The underlying model is assumed to expose three labels:
      0: Environmental
      1: Social
      2: Governance

    Args:
        sentences: List of input sentences.

    Returns:
        embeddings: Tensor of shape [num_sentences, 768]
        probabilities: Tensor of shape [num_sentences, 3]
    """
    if not sentences:
        # Return empty tensors with the correct number of dimensions
        return (
            torch.empty(0, 768, dtype=torch.float32),
            torch.empty(0, 3, dtype=torch.float32),
        )

    # Tokenize in a single batch for efficiency
    inputs = _tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    # Request hidden states so we can extract CLS embeddings
    outputs = _model(**inputs, output_hidden_states=True)

    # CLS embedding from the last hidden layer: [batch, seq_len, hidden] → [batch, hidden]
    last_hidden = outputs.hidden_states[-1]
    cls_embeddings = last_hidden[:, 0, :]  # shape: [num_sentences, 768]

    # ESG class probabilities
    logits = outputs.logits  # shape: [num_sentences, 3]
    probabilities = torch.softmax(logits, dim=-1)

    return cls_embeddings, probabilities

