
from arxiv_classifier.utils.get_model_config import get_model_config

def test_get_model_config_explicit_profile_tfidf():
    cfg = get_model_config(profile="tfidf", verbose=False)
    for k in ("hidden_layers","hidden_dim","learning_rate","dropout_rate","epochs","batch_size"):
        assert k in cfg
    assert cfg["profile"] == "tfidf"

def test_get_model_config_profile_overrides():
    cfg = get_model_config(profile="ngram", verbose=False)
    assert cfg["hidden_dim"] == 512
    assert cfg["dropout_rate"] == 0.55
