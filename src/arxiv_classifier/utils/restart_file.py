#!/usr/bin/python3

from pathlib import Path
import os

def restart_file(path):
    """
    Remove a file we plan to build if it already exists, so we can regenerate data without double dumping.
    """
    path = Path(path)
    try:
        os.remove(path)
        print(f"{path} has been removed. Regeneration of the data ready.")
    except FileNotFoundError:
        pass
