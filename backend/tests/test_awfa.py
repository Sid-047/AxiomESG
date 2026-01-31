from app.pipeline.awfa import apply_awfa


def test_awfa_dedup():
    sentences = {
        "E": ["Carbon emissions fell.", "Carbon emissions fell."],
        "S": [],
        "G": [],
    }
    weighted = apply_awfa(sentences)
    assert len(weighted) == 1
