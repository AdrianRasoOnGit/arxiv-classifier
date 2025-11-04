#!/usr/bin/python3
"""
Command Line Interface for the arxiv-classifier

Run preprocessing, vectorizations, and trainings from terminal commands.

Usage examples:

      arxiv-classifier preprocess
      arxiv-classifier vectorize --model tfidf
      arxiv-classifier train --model ngram
      arxiv-classifier all

"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable, List

# Preprocessing
from arxiv_classifier.preprocessing.json_to_parquet import convert_json_to_parquet
from arxiv_classifier.preprocessing.get_raw import get_raw
from arxiv_classifier.preprocessing.taxonomy_scraper import build_taxonomy
from arxiv_classifier.preprocessing.grab_labels import build_labels
from arxiv_classifier.preprocessing.raw_parquet_to_dataset import convert_parquet_to_dataset
from arxiv_classifier.preprocessing.dataset_to_clean_dataset import filter_dataset_labels
from arxiv_classifier.preprocessing.split_dataset import split_dataset

# Vectorizations
from arxiv_classifier.vectorizations.tfidf_vectorization import build_tfidf_vectorizer
from arxiv_classifier.vectorizations.ngram_vectorization import build_ngram_vectorizer
from arxiv_classifier.vectorizations.embeddings_vectorization import build_embeddings_vectorizer

# Trainings
from arxiv_classifier.trainings.train_tfidf import train_tfidf
from arxiv_classifier.trainings.train_ngram import train_ngram
from arxiv_classifier.trainings.train_embeddings import train_embeddings

# Logging helpers

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level = level,
        format= "[%(levelname)s] %(message)s",
    )

def log_stage(msg: str) -> None:
    logging.info(msg)

# STAGE WRAPPERS. Preprocessing

def stage_raw():
    """ Download the arxix dataset from Kaggle, just if it's missing """
    log_stage("Downloading raw dataset if missing.")
    get_raw()
    log_stage("Raw dataset ready.")

def stage_taxonomy():
    log_stage("Preparing arXiv repository taxonomy.")
    build_taxonomy()
    log_stage("Taxonomy extracted and saved.")

def stage_labels():
    log_stage("Extracting canonical labels from the arXiv taxonomy.")
    build_labels()
    log_stage("Labels saved.")

def stage_json_to_parquet():
    log_stage("Converting JSON raw into Parquet.")
    convert_json_to_parquet()
    log_stage("Raw Parquet created from the JSON raw.")

def stage_parquet_to_dataset():
    log_stage("Converting raw Parquet into Parquet dataset with labels text and label.")
    convert_parquet_to_dataset()
    log_stage("Parquet dataset created.")

def stage_clean():
    log_stage("Filtering dataset to canonical labels from the arXiv taxonomy.")
    filter_dataset_labels()
    log_stage("Clean dataset saved.")

def stage_split():
    log_stage("Splitting dataset into train and test splits.")
    split_dataset()
    log_stage("Split complete.")

def stage_preprocess():
    stage_taxonomy()
    stage_labels()
    stage_json_to_parquet()
    stage_parquet_to_dataset()
    stage_clean()
    stage_split()

# STAGE WRAPPERS. Vectorizations

def stage_vectorize(model: str):
    log_stage(f"Vectorizing with '{model}'.")
    if model == "tfidf":
        build_tfidf_vectorizer()
    elif model == "ngram":
        build_ngram_vectorizer()
    elif model == "embeddings":
        build_embeddings_vectorizer()
    else:
        raise ValueError(f"Unknown vectorization model {model}, try tfidf, ngram, or embeddings!")
    log_stage(f"Vectorization with {model} respresentation complete.")

# STAGE WRAPPERS. Trainings

def stage_train(model: str):
    log_stage(f"Training '{model}'.")
    if model == "tfidf":
        train_tfidf()
    elif model == "ngram":
        train_ngram()
    elif model == "embeddings":
        train_embeddings()

# CLI entrypoint

def main():
    parser = argparse.ArgumentParser(
        prog = "arxiv-classifier",
        description = "CLI for running arxiv-classifier pipeline.",
        formatter_class = argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-v", "--verbose",
        action = "count", default = 0,
        help = "Increase verbosity (-v for INFO, -vv for DEBUG)"
    )
    
    sub = parser.add_subparsers(dest = "command", required = True)

    # Preprocess tasks calls
    sub.add_parser("raw", help = "Download the arXiv dataset from Kaggle if missing.")
    sub.add_parser("taxonomy", help = "Download and save the arXiv taxonomy.")
    sub.add_parser("labels", help = "Extract and save canonical arXiv labels.")
    sub.add_parser("json-to-parquet", help = "Convert raw JSON data to raw Parquet data.")
    sub.add_parser("parquet-to-dataset", help = "Convert Parquet data to text and label dataset.")
    sub.add_parser("clean", help = "Filter dataset to keep only canonical labels.")
    sub.add_parser("split", help = "Split cleaned dataset into train and test splits.")

    # Preprocess stage call
    sub.add_parser("preprocess", help = "Run taxonomy, labels, json-to-parquet, parquet-to-dataset, clean, split.")

    # Vectorize stage call
    vectorize_parser = sub.add_parser("vectorize", help = "Vectorize dataset with a specified representation model.")
    vectorize_parser.add_argument(
        "--model", choices = ["tfidf", "ngram", "embeddings"], required = True,
        help = "Vectorization representation model."
    )

    # Train stage call
    train_parser = sub.add_parser("train", help = "Train a model using its corresponding vectors.")
    train_parser.add_argument(
        "--model", choices = ["tfidf", "ngram", "embeddings"], required = True,
        help = "Model to train."
    )

    # Full pipeline call
    sub.add_parser(
        "all",
        help = "Run full pipeline."
    )

    # Help

    help_parser = sub.add_parser("help", help = "Show CLI help message.")

    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Dispatch
    if args.command == "raw":
        stage_raw()

    elif args.command == "taxonomy":
        stage_taxonomy()

    elif args.command == "labels":
        stage_labels()

    elif args.command == "json-to-parquet":
        stage_json_to_parquet()

    elif args.command == "parquet-to-dataset":
        stage_parquet_to_dataset()

    elif args.command == "clean":
        stage_clean()

    elif args.command == "split":
        stage_split()

    elif args.command == "preprocess":
        stage_preprocess()

    elif args.command == "vectorize":
        stage_vectorize(args.model)

    elif args.command == "train":
        stage_train(args.model)

    elif args.command == "all":
        stage_raw()
        stage_taxonomy()
        stage_labels()
        stage_json_to_parquet()
        stage_parquet_to_dataset()
        stage_clean()
        stage_split()

        for m in ["tfidf", "ngram"]:
            stage_vectorize(m)
            stage_train(m)

    # CLI help call
    
    elif args.command == "help":
        log_stage("""

arxiv-classifier CLI Help Space
-------------------------------
        
This is the Command-Line Interface for the arxiv-classifier project. You can run any stage of the project individually, execute the entire workflow, or particular scripts. Check the inventory below to see which operations of the repo are available.

Usage:
arxiv-classifier <command>

Commands:
    raw                  Download the arXiv raw dataset from Kaggle, if missing.
    taxonomy             Download and save the arXiv taxonomy.
    labels               Extract canonical arXiv labels.
    json-to-parquet      Convert raw JSON data to Parquet format.
    parquet-to-dataset   Convert Parquet files into a dataset.
    clean                Filter dataset to canonical labels.
    split                Split cleaned dataset into train/test sets.
    training             Train all available models using their corresponding vectors.
    vectorize <model>    Vectorize using models such as tfidf, ngram, or embeddings.
    train <model>        Train using a single model, tfidf, ngram, or embeddings.
    preprocess           Run json to parquet, to dataset, to clean steps.
    all                  Run the full preprocessing + training pipeline.
    help                 Show this message.

Examples:
    arxiv-classifier -vv preprocessing
    arxiv-classifier -v all
    arxiv-classifier -vv train --model tfidf
        
        """)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
