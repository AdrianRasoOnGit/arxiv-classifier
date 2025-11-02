#!/usr/bin/python3

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
src_path = root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print(f"__file__: {__file__}")
print(f"root: {root}")
print(f"src_path: {src_path}")

# Directories
data_dir = root / "data"
scripts_dir = root / "scripts"
models_dir = root / "models"
reports_dir = root/ "reports"
# legacy folder: training_dir = root / "training"

# Data subdirectories
raw_subdir = data_dir / "raw"
processed_subdir = data_dir / "processed"
splits_subdir = processed_subdir / "splits"
meta_subdir = data_dir / "meta"

# data/raw
raw_json_path = raw_subdir / "raw.json"
raw_parquet_path = raw_subdir / "raw.parquet"

# data/meta
taxonomy_path = meta_subdir / "taxonomy.json"
labels_path = meta_subdir / "labels.txt"
ds_labels_path = meta_subdir / "ds_labels.txt"

# data/processed
dataset_path = processed_subdir / "dataset.parquet"
clean_dataset_path = processed_subdir / "clean_dataset.parquet"
vectors_subdir = processed_subdir / "vectors"

# data/processed/vectors
tfidf_vectors_subdir = vectors_subdir / "tfidf"
ngram_vectors_subdir = vectors_subdir / "ngram"
embeddings_vectors_subdir = vectors_subdir / "embeddings"

# data/processed/vectors/tfidf
X_train_tfidf_path = tfidf_vectors_subdir / "X_train.npz"
y_train_tfidf_path = tfidf_vectors_subdir / "y_train.npy"
X_test_tfidf_path = tfidf_vectors_subdir / "X_test.npz"
y_test_tfidf_path = tfidf_vectors_subdir / "y_test.npy"

# data/processed/vectors/ngram
X_train_ngram_path = ngram_vectors_subdir / "X_train.npz"
y_train_ngram_path = ngram_vectors_subdir / "y_train.npy"
X_test_ngram_path = ngram_vectors_subdir / "X_test.npz"
y_test_ngram_path = ngram_vectors_subdir / "y_test.npy"

# data/processed/vectors/embeddings
vectorizer_embeddings_path = embeddings_vectors_subdir / "embeddings_vectorizer.joblib"
X_train_embeddings_path = embeddings_vectors_subdir / "X_train.npz"
y_train_embeddings_path = embeddings_vectors_subdir / "y_train.npy"
X_test_embeddings_path = embeddings_vectors_subdir / "X_test.npz"
y_test_embeddings_path = embeddings_vectors_subdir / "y_test.npy"

# data/splits
train_path = splits_subdir / "train.parquet"
test_path = splits_subdir / "test.parquet"

# models/
with_tfidf_subdir = models_dir / "with_tfidf"
with_ngram_subdir = models_dir / "with_ngram"
with_embeddings_subdir = models_dir / "with_embeddings"

# models/with_tfidf
vectorizer_tfidf_path = with_tfidf_subdir / "tfidf_vectorizer.joblib"
weights_tfidf_path = with_tfidf_subdir / "weights_tfidf.npz"

# models/with_ngram
vectorizer_ngram_path = with_ngram_subdir / "ngram_vectorizer.joblib"
weights_ngram_path = with_ngram_subdir / "weights_ngram.npz"

# models/with_embeddings
vectorizer_embeddings_path = with_embeddings_subdir / "embeddings_vectorizer.joblib"
weights_embeddings_path = with_embeddings_subdir / "weights_embeddings.npz"

# Reports subdirectories
charts_subdir = reports_dir / "charts"
metrics_subdir = reports_dir / "metrics"

# Reports summary
summary_path = reports_dir / "summary.txt"

# reports/metrics
tfidf_classification_report_path = metrics_subdir / "tfidf_classification_report.csv"
ngram_classification_report_path = metrics_subdir / "ngram_classification_report.csv"
embeddings_classification_report_path = metrics_subdir / "embeddings_classification_report.csv"

# reports/charts
tfidf_training_loss_path = charts_subdir / "tfidf_training_loss.png"
tfidf_confusion_matrix_path = charts_subdir / "tfidf_confusion_matrix.png"
ngram_training_loss_path = charts_subdir / "ngram_training_loss.png"
ngram_confusion_matrix_path = charts_subdir / "ngram_confusion_matrix.png"
embeddings_training_loss_path = charts_subdir / "embeddings_training_loss.png"
embeddings_confusion_matrix_path = charts_subdir / "embeddings_confusion_matrix.png"

# Display of paths
def display_paths():
    print("Current path definitions for directories: ")
    print("Project root: ", root)
    print("Data directory: ", data_dir)
    print("Scripts directory: ", scripts_dir)

# All paths
__all__ = [
    "root", "data_dir", "raw_json_path", "raw_parquet_path",
    "taxonomy_path", "labels_path",
    "dataset_path", "clean_dataset_path",
    "train_path", "test_path", "display_paths",
]    
