import pytest

from csmr.config import load_config


def test_defaults():
    assert load_config()["tau_e"] == 0.6


def test_dotted_overrides_typed():
    cfg = load_config(overrides=["lambda_c=0.3", "ablation.use_dependency=false"])
    assert cfg["lambda_c"] == 0.3
    assert cfg["ablation"]["use_dependency"] is False
    assert cfg["ablation"]["use_evidence"] is True  # untouched siblings survive


def test_cache_isolated_from_mutation():
    load_config()["tau_e"] = 99
    assert load_config()["tau_e"] == 0.6


def test_bad_override():
    with pytest.raises(ValueError):
        load_config(overrides=["lambda_c"])
