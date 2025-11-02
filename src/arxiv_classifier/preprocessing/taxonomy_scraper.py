#!/usr/bin/python3

from bs4 import BeautifulSoup
import requests
import json
from pathlib import Path
from arxiv_classifier import taxonomy_path

def extract_taxonomy(url: str = "https://arxiv.org/category_taxonomy") -> dict:
    html = requests.get(url).text
    contents = BeautifulSoup(html, "html.parser")

    taxonomy = {}

    # Each top-level group has <h2 class="accordion-head">
    for group_head in contents.select("h2.accordion-head"):
        group_name = group_head.get_text(strip = True)
        taxonomy[group_name] = []

        # Its sibling <div class="accordion-body"> contains the subject categories
        group_body = group_head.find_next_sibling("div", class_ = "accordion-body")
        if not group_body:
            continue

        # Each category is within a <div class="columns divided"> block
        for cat_div in group_body.select("div.columns.divided"):
            code_tag = cat_div.select_one("h4")
            desc_tag = cat_div.select_one("p")

            if not code_tag:
                continue

        # Extract code (like 'cs.AI') and name (like 'Artificial Intelligence')
            code_text = code_tag.contents[0].strip() if code_tag.contents else ""
            name_span = code_tag.find("span")
            name_text = ""
            if name_span:
                name_text = name_span.get_text(strip = True).strip("()")

            desc_text = desc_tag.get_text(strip = True) if desc_tag else ""

            taxonomy[group_name].append({
                "code": code_text,
                "name": name_text,
                "description": desc_text
            })

    return taxonomy


def save_taxonomy(taxonomy: dict, path: Path = taxonomy_path) -> None:

    with open(path, "w", encoding = "utf-8") as f:
        json.dump(taxonomy, f, indent = 2, ensure_ascii = False)

    print(f"arxiv papers taxonomy has been extracted and saved in taxonomy.json, inside the data/meta/ folder, with {sum(len(v) for v in taxonomy.values())} categories total, of the 155 actual categories (as manually checked in October 2025).")

def build_taxonomy():

    taxonomy = extract_taxonomy()
    save_taxonomy(taxonomy)
