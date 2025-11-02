#!/usr/bin/python3

from pathlib import Path
import numpy as np
from scipy.sparse import load_npz, issparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from ..paths import(X_train_ngram_path,
                  y_train_ngram_path,
                  X_test_ngram_path,
                  y_test_ngram_path,
                  weights_ngram_path,
                  ngram_classification_report_path,
                  ngram_confusion_matrix_path,
                  ngram_training_loss_path,
                  summary_path,
                  )

from ..utils.get_model_config import get_model_config
from ..utils.load_training_data import load_training_data
from ..utils.log_training_summary import log_training_summary

from ..models.fmlp import NeuralNetwork

def train_ngram():
    
    # Load configuration and data
    config = get_model_config()
    X_train, y_train, X_test, y_test, encoder = load_training_data(config)

    # Build model
    model = NeuralNetwork.setup(X_train.shape[1], len(encoder.classes_), config)

    # Train
    print("N-gram model training started.")
    losses, accs = model.train(X_train, y_train, config)

    # Evaluate
    print("Evaluation of the model.")
    cache = model.forward(X_test, dropout = False)
    y_pred_probs = cache[f"A{model.L}"]
    y_pred = np.argmax(y_pred_probs, axis = 1)
    acc = np.mean(y_pred == y_test)
    print(f"Test accuracy: {acc:.4f}")

    # Plot training metrics
    plt.figure(figsize = (6, 4))
    plt.plot(losses, label = "loss")
    plt.plot(accs, label = "Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("N-gram Training Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ngram_training_loss_path, dpi = 200)
    plt.close()

    # Classification report
    all_labels = np.arange(len(encoder.classes_))
    report = classification_report(
        y_test,
        y_pred,
        labels = all_labels,
        target_names = encoder.classes_,
        zero_division = 0,
        output_dict = True,
    )
    pd.DataFrame(report).transpose().to_csv(
        ngram_classification_report_path, index = True, encoding = "utf-8"
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels = all_labels)

    if cm.shape[0] < len(all_labels):
        print(
            f" Confusion matrix {len(all_labels) - cm.shape[0]} labels. "
            f"Padding to full {len(all_labels)}x{len(all_labels)}."
        )
        full_cm = np.zeros((len(all_labels), len(all_labels)), dtype = int)
        full_cm[: cm.shape[0], : cm.shape[1]] = cm
        cm = full_cm

    top_k = 20
    label_counts = np.bincount(y_test, minlength = len(encoder.classes_))
    top_labels = np.argsort(label_counts)[-top_k:]

    cm_top = cm[np.ix_(top_labels, top_labels)]
    plt.figure(figsize = (8, 6))
    sns.heatmap(
        cm_top,
        cmap = "Blues",
        xticklabels = encoder.classes_[top_labels],
        yticklabels = encoder.classes_[top_labels],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("N-gram NN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(ngram_confusion_matrix_path, dpi = 200)
    plt.close()

    # Log summary
    log_training_summary(summary_path, acc, config)
    print("Reporsts saved in /reports.")

    # Save weights
    np.savez(
        weights_ngram_path,
        **model.params
    )
    print("Model weights saved in models/.")

    return model, encoder, acc
