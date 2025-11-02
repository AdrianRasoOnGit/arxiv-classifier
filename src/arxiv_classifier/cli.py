#!/usr/bin/python3

import argparse

from arxiv_classifier.preprocessing.json_to_parquet import convert_json_to_parquet
from arxiv_classifier.preprocessing.get_raw import get_raw
from arxiv_classifier.preprocessing.taxonomy_scraper import build_taxonomy
from arxiv_classifier.preprocessing.grab_labels import build_labels
from arxiv_classifier.preprocessing.raw_parquet_to_dataset import convert_parquet_to_dataset
from arxiv_classifier.preprocessing.dataset_to_clean_dataset import filter_dataset_labels
from arxiv_classifier.preprocessing.split_dataset import split_dataset
from arxiv_classifier.vectorizations.tfidf_vectorization import build_tfidf_vectorizer
from arxiv_classifier.vectorizations.ngram_vectorization import build_ngram_vectorizer
from arxiv_classifier.vectorizations.embeddings_vectorization import build_embeddings_vectorizer
from arxiv_classifier.trainings.train_tfidf import train_tfidf
from arxiv_classifier.trainings.train_ngram import train_ngram
from arxiv_classifier.trainings.train_embeddings import train_embeddings

def main():
    parser = argparse.ArgumentParser(
        prog = "arxiv-classifier",
        description = "Interface for managing processes related to generating the project."
    )

    sub = parser.add_subparsers(dest = "command", required = True)

    # Preprocessing commands
    sub.add_parser("raw", help = "Download the arXiv dataset from Kaggle.")
    sub.add_parser("taxonomy", help = "Download and save the arXiv taxonomy.")
    sub.add_parser("labels", help = "Extract and save canonical arXiv labels from taxonomy.")
    sub.add_parser("json-to-parquet", help = "Convert raw JSON data to Parquet format.")
    sub.add_parser("parquet-to-dataset", help = "Convert Parquet data to text + label dataset.")
    sub.add_parser("clean", help = "Filter dataset to keep only canonical labels.")
    sub.add_parser("split", help = "Split cleaned dataset into train/test sets.")
    sub.add_parser("preprocess", help = "Run steps from original Kaggle raw dataset to this project dataset.")
    sub.add_parser("vectorization", help = "Build all kinds of vectors of the project.")
    sub.add_parser("training", help = "Perform training on all models and vectors.")

    # Vectorization commands
    vectorize_parser = sub.add_parser("vectorize", help = "Vectorize dataset using a specified method.")
    vectorize_parser.add_argument(
        "model",
        choices = ["tfidf", "ngram", "embeddings"],
        help = "Vectors methods available."
    )
    

    # Train commands
    train_parser = sub.add_parser("train", help = "Train a model.")
    train_parser.add_argument(
        "model",
        choices = ["tfidf", "ngram", "embeddings"],
        help = "Models to train."
    )
    
    sub.add_parser("all", help = "Run the complete project, from the preprocessing pipeline, to the vectorization and training steps.")
    
    help_parser = sub.add_parser("help", help = "This is the Command-Line Interface of the arxiv-classifier project. Here, you can run any part of the repo, or perform a complete run of all the elements contained, reconstructing all the data, vectorizing and performing the trainings. The functions you can access from here are: taxonomy, labels, json to parquet, parquet to dataset, clean, split, preprocess (which is the complete preprocess pipeline), tfidf, train-tfidf, preprocessing, and all.")
                                 
    args = parser.parse_args()

    # Dispatch logic

    # Complete project stages
    
    if args.command == "preprocessing":
        build_taxonomy()
        build_labels()
        convert_json_to_parquet()
        convert_parquet_to_dataset()
        filter_dataset_labels()
        split_dataset()

    elif args.command == "vectorization":
        build_tfidf_vectorizer()
        build_ngram_vectorizer()
        build_embeddings_vectorizer()

    elif args.command == "training":
        train_tfidf()
        train_ngram()
        train_embeddings()

    elif args.command == "all":
        get_raw()
        build_taxonomy()
        build_labels()
        convert_json_to_parquet()
        convert_parquet_to_dataset()
        filter_dataset_labels()
        split_dataset()
        build_tfidf_vectorizer()
        train_tfidf()
        build_ngram_vectorizer()
        train_ngram()
        build_embeddings_vectorizer()
        train_embeddings()
        
    # Preprocessing stages calls
    
    elif args.command == "raw":
        get_raw()
    
    elif args.command == "taxonomy":
        build_taxonomy()

    elif args.command == "labels":
        build_labels()

    elif args.command == "json-to-parquet":
        convert_json_to_parquet()

    elif args.command == "parquet-to-dataset":
        convert_parquet_to_dataset()

    elif args.command == "clean":
        filter_dataset_labels()

    elif args.command == "split":
        split_dataset()

    # Vectorization stages calls
    
    elif args.command == "vectorize":
        
        if args.model == "tfidf":
            build_tfidf_vectorizer()

        elif args.model == "ngram":
            build_ngram_vectorizer()

        elif args.model == "embeddings":
            build_embeddings_vectorizer()

    # Train stages calls
    
    elif args.command == "train":
        
        if args.model == "tfidf":
            train_tfidf()
            
        elif args.model == "ngram":
            train_ngram()
            
        elif args.model == "embeddings":
            train_embeddings()

    # CLI help call
    
    elif args.command == "help":
        print("""

arxiv-classifier CLI Help Space
-------------------------------
        
This is the Command-Line Interface for the arxiv-classifier project. You can run any stage of the project individually, execute the entire workflor, or particular scripts. Check the inventory below to see which operations of the repo are available.

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
    arxiv-classifier preprocessing
    arxiv-classifier all
    arxiv-classifier train tfidf
        
        """)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
