#!/usr/bin/python3

from pathlib import Path
import shutil
import kagglehub
from arxiv_classifier import paths

def get_raw():
    dst = raw_json_path

    if dst.exists():
        print("Raw data present at: {dest}. Skipping download.")
        return dst

    print("Downloading arXiv dataset from Kaggle. This is the raw dataset we will use in the project.")
    kaggle_path = Path(kagglehub.dataset_download("Cornell-University/arxiv"))

    kaggle_path.rename(dst)

    print(f"Dataset ready at: {dst}.")
    return dst

if __name__ == "__main__":
    get_raw()
