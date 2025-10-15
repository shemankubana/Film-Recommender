"""
Preprocess the CSV dataset and tokenize using GPT-2 tokenizer.
Outputs:
- data/cinema_tmdb_clean.json
- data/tokenized_dataset.npz
Follows assignment preprocessing steps: lowercase, remove urls/html/special chars, tokenization, padding/truncation, missing value handling.
"""

import re
import os
import csv
import json
import argparse
import numpy as np
from transformers import GPT2Tokenizer
from tqdm import tqdm

# Regex patterns
url_re = re.compile(r'https?://\S+|www\.\S+')
html_re = re.compile(r'<.*?>')
special_re = re.compile(r'[^0-9A-Za-z\-\.,\'\"\?\!\:\;\s()]')
multi_space = re.compile(r'\s+')

PAD_TOKEN = "<|pad|>"

def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = text.lower()
    text = url_re.sub('', text)
    text = html_re.sub('', text)
    text = special_re.sub('', text)
    text = multi_space.sub(' ', text)
    return text.strip()

def load_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def main(raw_csv="data/cinema_tmdb_dataset.csv", clean_json="data/cinema_tmdb_clean.json",
         tokenized_save="data/tokenized_dataset.npz", model_name="gpt2", max_length=128):
    os.makedirs("data", exist_ok=True)
    raw = load_csv(raw_csv)
    cleaned = []
    for r in raw:
        user = clean_text(r.get("user_input",""))
        bot = clean_text(r.get("bot_response",""))
        if user == "" or bot == "":
            # Handling missing values: skip (documented)
            continue
        cleaned.append({"id": r.get("id",""), "user_input": user, "bot_response": bot})

    # Save cleaned JSON
    with open(clean_json, "w", encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    # Tokenize using GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})

    input_ids_list = []
    attn_list = []
    labels_list = []

    for pair in tqdm(cleaned, desc="Tokenizing"):
        prompt = f"User: {pair['user_input']}  Bot:"
        target = f" {pair['bot_response']}"
        full = prompt + target
        enc_full = tokenizer(full, truncation=True, max_length=max_length, padding='max_length')
        input_ids = np.array(enc_full["input_ids"], dtype=np.int32)
        attn = np.array(enc_full["attention_mask"], dtype=np.int32)

        # compute prompt length in tokens (using tokenizer on prompt only)
        enc_prompt = tokenizer(prompt, truncation=True, max_length=max_length, padding='max_length')
        prompt_len = sum(1 for t in enc_prompt["input_ids"] if t != tokenizer.pad_token_id)

        labels = input_ids.copy()
        labels[:prompt_len] = -100  # ignore prompt tokens in loss

        input_ids_list.append(input_ids)
        attn_list.append(attn)
        labels_list.append(labels)

    arr = {
        "input_ids": np.stack(input_ids_list),
        "attention_mask": np.stack(attn_list),
        "labels": np.stack(labels_list)
    }
    np.savez_compressed(tokenized_save, **arr)
    print("Saved tokenized dataset to:", tokenized_save)
    print("Dataset size:", arr["input_ids"].shape[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", default="data/cinema_tmdb_dataset.csv")
    parser.add_argument("--clean_json", default="data/cinema_tmdb_clean.json")
    parser.add_argument("--tokenized_save", default="data/tokenized_dataset.npz")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--model_name", default="gpt2")
    args = parser.parse_args()
    main(raw_csv=args.raw_csv, clean_json=args.clean_json, tokenized_save=args.tokenized_save,
         max_length=args.max_length, model_name=args.model_name)
