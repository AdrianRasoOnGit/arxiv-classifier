#!/usr/bin/python3
"""
Extract the set of labels from the arXiv taxonomy

This module makes it possible to obtain a set containing the categories recognized by the arXiv taxonomy. In this way, we will later build the y vector or label vector.
"""
import json
from pathlib import Path
from arxiv_classifier import taxonomy_path, labels_path, restart_file

def extract_labels(taxonomy_json: Path = taxonomy_path) -> list[str]:
    """
    Extract all category codes from the taxonomy file.

    Args:
        taxonomy_json (Path): Path to the taxonomy JSON file, that we scrapped from the official arXiv taxonomy site.

    Returns:
        list[str]: List of all category codes, sorted.
    """
    
    with open(taxonomy_json, "r", encoding = "utf-8") as f:
        taxonomy = json.load(f)

    labels = [cat["code"] for cats in taxonomy.values() for cat in cats if "code" in cat]
    
    return sorted(labels)


def save_labels(labels: list[str], output_path: Path = labels_path) -> None:
    """
    Save the list of labels to a text file.

    """
    restart_file(output_path)
    
    with open(output_path, "w", encoding = "utf-8") as f:
        f.write("\n".join(labels))

    print(f"{len(labels)} labels (from 155 arXiv categories) saved to {output_path}.")


def build_labels() -> None:

    labels = extract_labels()
    save_labels(labels)
