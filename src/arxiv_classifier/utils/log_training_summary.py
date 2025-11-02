#!/usr/bin/python3

import inspect
from datetime import datetime

from ..paths import summary_path


def log_training_summary(summary_path, acc, config, verbose = True):

    caller = inspect.stack()[1].function.lower()

    if "tfidf" in caller:
        vector_type = "TF-IDF"
    elif "ngram" in caller:
        vector_type = "N-gram"
    elif "embedding" in caller or "embeddings" in caller:
        vector_type = "Embeddings"
    else:
        vector_type = "Unknown"

    # Timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y %H:$M:%S")

    # Append results
    
    with open(summary_path, "a", encoding = "utf-8") as f:
        f.write("\n" + "-"*60)
        # f.write(f"Trial: {trial_number}")
        # trial_number += 1
        f.write(f"\n[{timestamp}] {vector_type} Forward MultiLayer Perceptron")
        f.write(f"\nAccuracy: {acc:.4f}")
        f.write("\nHyperparameters: ")
        for k, v in config.items():
            f.write(f"\n   {k}: {v}")
        f.write("\n")

    print("Reports saved in /reports.")
    
