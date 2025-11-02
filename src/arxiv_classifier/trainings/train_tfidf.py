#!/usr/bin/python3

from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from ..paths import(X_train_tfidf_path,
                  y_train_tfidf_path,
                  X_test_tfidf_path,
                  y_test_tfidf_path,
                  weights_tfidf_path,
                  tfidf_classification_report_path,
                  tfidf_confusion_matrix_path,
                  tfidf_training_loss_path,
                  summary_path
                  )

from ..utils.get_model_config import get_model_config
from ..utils.log_training_summary import log_training_summary
from ..utils.load_training_data import load_training_data
from ..models.fmlp import NeuralNetwork

def train_tfidf():
    
    # Load training, validation and test data
    config = get_model_config()
    X_train, y_train, X_test, y_test, encoder = load_training_data(config)

    # Build model
    model = NeuralNetwork.setup(X_train.shape[1], len(encoder.classes_), config)

    # Train (and get some metrics for the evaluation)
    print("TF-IDF model training started.")
    losses, accs = model.train(X_train, y_train, config)

    # Evaluate
    print("Evaluation of the model.")
    cache = model.forward(X_test, dropout = False)
    y_pred_probs = cache[f"A{model.L}"]
    y_pred = np.argmax(y_pred_probs, axis = 1)
    acc = np.mean(y_pred == y_test)
    print(f"Test accuracy: {acc:.4f}")

    # Create reports

    # Plot training curves
    plt.figure(figsize = (6, 4))
    plt.plot(losses, label = "Loss")
    plt.plot(accs, label = "Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("TF-IDF Training Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(tfidf_training_loss_path, dpi = 200)
    plt.close()

    # Classification report
    all_labels = np.arange(len(encoder.classes_))
    
    labels_present = np.unique(y_test)
    report = classification_report(
        y_test,
        y_pred,
        labels = all_labels,
        target_names = encoder.classes_,
        zero_division = 0,
        output_dict = True
        )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(tfidf_classification_report_path, index = True, encoding = "utf-8")
    
    # with open(tfidf_classification_report_path, "w") as f:
        # f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels = all_labels)

    if cm.shape[0] < len(all_labels):
        print(f" Confusion matrix {len(all_labels) - cm.shape[0]} labels. Padding to full {len(all_labels)}x{len(all_labels)}.")
        full_cm = np.zeros((len(all_labels), len(all_labels)), dtype = int)
        full_cm[:cm.shape[0], :cm.shape[1]] = cm
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
    plt.title("TF-IDF NN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(tfidf_confusion_matrix_path, dpi = 200)
    plt.close()

    # Summary of accuracies, with hyperparameters so we can know how we got that performance
    log_training_summary(summary_path, acc, config)

    print("Reports saved in /reports.")

    # Save weights
    np.savez(
        weights_tfidf_path,
        **model.params
    )
    print("Model weights saved in models/.")

    return model, encoder, acc
