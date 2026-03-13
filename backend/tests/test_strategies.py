"""Tests for the strategy registry."""

from app.pipeline.strategies import get_strategy, list_strategies, STRATEGY_META


def test_heuristic_registered():
    """The heuristic strategy should be immediately available."""
    fn = get_strategy("heuristic")
    assert callable(fn)


def test_heuristic_produces_output():
    """The heuristic strategy should produce sorted tuples."""
    fn = get_strategy("heuristic")
    result = fn({
        "E": ["We reduced carbon emissions by 12%."],
        "S": [],
        "G": [],
    })
    assert len(result) == 1
    cat, sent, weight = result[0]
    assert cat == "E"
    assert weight > 0


def test_list_strategies_contains_heuristic():
    """list_strategies should include the heuristic entry."""
    strategies = list_strategies()
    keys = [s["key"] for s in strategies]
    assert "heuristic" in keys


def test_unknown_strategy_raises():
    """Requesting an unknown strategy should raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Unknown algorithm"):
        get_strategy("nonexistent_algorithm")


def test_all_strategy_meta_keys():
    """All STRATEGY_META keys should have label and description."""
    for key, meta in STRATEGY_META.items():
        assert "label" in meta, f"Missing label for {key}"
        assert "description" in meta, f"Missing description for {key}"
