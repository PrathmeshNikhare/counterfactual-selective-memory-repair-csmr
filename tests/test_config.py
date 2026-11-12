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


def test_overlay_deep_merges_and_set_wins(tmp_path):
    ov = tmp_path / "tuned.yaml"
    ov.write_text("# comment\ntau_d: 0.4\nlambda_c: 0.25\nablation: {use_minimality: false}\n")
    cfg = load_config(overrides=["lambda_c=0.1"], overlay=str(ov))
    assert cfg["tau_d"] == 0.4 and cfg["lambda_c"] == 0.1  # --set beats the overlay
    assert cfg["ablation"]["use_minimality"] is False and cfg["ablation"]["use_dependency"] is True
    assert load_config()["tau_d"] == 0.5  # no overlay, no change
    with pytest.raises(FileNotFoundError):
        load_config(overlay=str(tmp_path / "missing.yaml"))


def test_bad_override():
    with pytest.raises(ValueError):
        load_config(overrides=["lambda_c"])
