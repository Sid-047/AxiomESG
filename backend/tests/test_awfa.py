from app.pipeline.awfa import apply_awfa


def test_awfa_dedup():
    sentences = {
        "E": ["Carbon emissions fell.", "Carbon emissions fell."],
        "S": [],
        "G": [],
    }
    weighted = apply_awfa(sentences)
    assert len(weighted) == 1


def test_awfa_sorting():
    """Verify that longer, more keyword-dense sentences score higher."""
    sentences = {
        "E": [
            "Short.",
            "We reduced carbon emissions significantly across all facilities in 2024.",
        ],
        "S": [],
        "G": [],
    }
    weighted = apply_awfa(sentences)
    assert len(weighted) == 2
    # The longer sentence with 'carbon' and 'emissions' should score higher
    assert weighted[0][1].startswith("We reduced")
