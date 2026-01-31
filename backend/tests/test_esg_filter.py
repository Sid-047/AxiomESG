from app.core.config import Settings
from app.pipeline.esg_filter import filter_esg_sentences


def test_esg_filter_basic():
    settings = Settings()
    text = "We reduced carbon emissions by 12%. Employee safety improved."
    result = filter_esg_sentences(text, settings)
    assert any("carbon" in s.lower() for s in result["E"])
    assert any("safety" in s.lower() for s in result["S"])
