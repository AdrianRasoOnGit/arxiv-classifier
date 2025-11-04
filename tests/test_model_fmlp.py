
import numpy as np
from arxiv_classifier.models.fmlp import NeuralNetwork 

def test_setup_and_forward_shapes():
    cfg = {"hidden_layers": 2, "hidden_dim": 8, "learning_rate": 1e-3, "dropout_rate": 0.0}
    model = NeuralNetwork.setup(input_dim=5, output_dim=3, config=cfg)

    X = np.random.RandomState(0).randn(4, 5).astype(np.float32)
    cache = model.forward(X, dropout=False)
    probs = cache[f"A{model.L}"]
    assert probs.shape == (4, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), rtol=1e-5, atol=1e-5)

def test_train_returns_curves_and_updates():
    cfg = {"hidden_layers": 1, "hidden_dim": 6, "learning_rate": 1e-2, "dropout_rate": 0.0, "lam": 0.0,
           "epochs": 3, "batch_size": 2}
    model = NeuralNetwork.setup(5, 2, cfg)
    X = np.random.RandomState(1).randn(6,5).astype(np.float32)
    y = np.array([0,1,0,1,0,1], dtype=int)
    losses, accs = model.train(X, y, cfg)
    assert len(losses) == 3 and len(accs) == 3
