#!/usr/bin/python3

from pathlib import Path
import numpy as np

from ..paths import(
    X_train_embeddings_path,
    y_train_embeddings_path,
    X_test_embeddings_path,
    y_test_embeddings_path,
    weights_embeddings_path,
)

from ..config.models_config import FMLP_CONFIG
from ..utils.load_training_data import load_training_data
from ..models.fmlp import NeuralNetwork

def train_embeddings():

    # Load the data
    X_train, y_train, X_test, y_test, encoder = load_training_data(config = FMLP_CONFIG)

    # Build model
    model = NeuralNetwork.setup(X_train.shape[1], len(encoder.classes_), conig = FMLP_CONFIG)

    # Train
    print("Embeddings training started.")
    model.train(X_train, y_train, config = FMLP_CONFIG)

    # Evaluate
    print("Evaluation of the model.")
    cache = model.forward(X_test, dropout = False)
    y_pred_probs = cache[f"A{model.L}"]
    y_pred = np.argmax(y_pred_probs, axis = 1)
    acc = np.mean(y_pred == y_test)
    print(f"Test accuracy: {acc:.4f}")

    # Save weights
    np.savez(
        weights_embeddings_path,
        **model.params
    )
    print("Model weights saved in models/.")

    return model, encoder, acc
