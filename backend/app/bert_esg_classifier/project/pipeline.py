from pathlib import Path
from typing import List

import torch
from torch import Tensor

from text_processing import split_sentences
from bert_processor import process_sentences
from fusion.mean_fusion import MeanFusion
from fusion.static_fusion import StaticFusion
from fusion.awfa_v1 import AWFAv1
from fusion.awfa_v2 import AWFAv2
from evaluation import evaluate_fusion_methods


def _load_report_text() -> str:
    """
    Load raw ESG report text.

    In a real system this would come from a file upload, storage,
    or another upstream component. For now we load a local sample
    if present, otherwise fall back to a short inline example.
    """
    this_dir = Path(__file__).resolve().parent
    sample_path = this_dir / "sample_report.txt"

    if sample_path.exists():
        return sample_path.read_text(encoding="utf-8")

    # Minimal inline example so the pipeline can run end-to-end.
    return (
        "Our company reduced carbon emissions by 20% this year. "
        "We improved workplace safety and employee well-being. "
        "The board strengthened governance and compliance oversight."
    )


def _build_signals(
    embeddings: Tensor, probabilities: Tensor
) -> List[Tensor]:
    """
    Construct the three core signals expected by fusion models:

      Signal 1: BERT embeddings          → [num_sentences, 768]
      Signal 2: ESG probability vectors  → projected to [num_sentences, 768]
      Signal 3: classification confidence → tiled to [num_sentences, 768]

    All signals are converted to tensors with a shared feature_dim
    so that fusion models can treat them uniformly.
    """
    if embeddings.numel() == 0:
        # Return empty tensors with the right feature dimension
        num_sentences = 0
        feature_dim = 768
        empty = torch.empty(num_sentences, feature_dim, dtype=torch.float32)
        return [empty, empty.clone(), empty.clone()]

    num_sentences, feature_dim = embeddings.shape

    if feature_dim != 768:
        raise ValueError(f"Expected embedding size 768, got {feature_dim} instead.")

    # Signal 1: raw CLS embeddings
    embedding_tensor = embeddings

    # Signal 2: probabilities projected to 768 dims by simple tiling:
    # shape [num_sentences, 3] → [num_sentences, 3 * 256] = [num_sentences, 768]
    if probabilities.shape[-1] != 3:
        raise ValueError(
            f"Expected 3-way ESG probabilities, got {probabilities.shape[-1]} classes."
        )
    probability_tensor = probabilities.repeat(1, 256)

    # Signal 3: confidence = max(probabilities) per sentence,
    # then repeat to match the feature dimension.
    confidence = probabilities.max(dim=-1).values  # [num_sentences]
    confidence_tensor = confidence.unsqueeze(-1).repeat(1, feature_dim)

    return [embedding_tensor, probability_tensor, confidence_tensor]


def run_pipeline() -> None:
    """
    End-to-end ESG text analysis pipeline:

      1. Load raw report text
      2. Split into sentences
      3. Run BERT to obtain embeddings and ESG probabilities
      4. Construct fusion signals
      5. Run all four fusion strategies
      6. Compare and print results
    """
    raw_text = _load_report_text()

    # 1) Split into sentences
    sentences = split_sentences(raw_text)
    print(f"Number of sentences: {len(sentences)}")

    if not sentences:
        print("No sentences found; nothing to fuse.")
        return

    # 2) BERT processing (batch)
    embeddings, probabilities = process_sentences(sentences)

    # 3) Signal construction
    signals = _build_signals(embeddings, probabilities)

    # 4) Instantiate fusion models
    mean_model = MeanFusion()
    static_model = StaticFusion()

    num_signals = len(signals)
    feature_dim = signals[0].shape[-1]

    awfa_v1_model = AWFAv1(num_signals=num_signals, feature_dim=feature_dim)
    awfa_v2_model = AWFAv2(num_signals=num_signals, feature_dim=feature_dim)

    # 5) Apply fusion methods
    mean_fused, mean_weights = mean_model([embeddings, probabilities])
    static_fused, static_weights = static_model([embeddings, probabilities])

    # AWFA models consume the full set of signals
    awfa_v1_fused, awfa_v1_weights = awfa_v1_model(signals)
    awfa_v2_fused, awfa_v2_weights = awfa_v2_model(signals)

    # 6) Evaluation and comparison
    evaluate_fusion_methods(
        sentences=sentences,
        mean_result=(mean_fused, mean_weights),
        static_result=(static_fused, static_weights),
        awfa_v1_result=(awfa_v1_fused, awfa_v1_weights),
        awfa_v2_result=(awfa_v2_fused, awfa_v2_weights),
    )


if __name__ == "__main__":
    run_pipeline()

