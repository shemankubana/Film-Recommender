"""
Preprocess the CSV dataset and tokenize using GPT-2 tokenizer.
NOW WITH TRAIN/VALIDATION SPLIT
Outputs:
- data/cinema_tmdb_clean.json
- data/tokenized_train.npz
- data/tokenized_val.npz
Follows assignment preprocessing steps: lowercase, remove urls/html/special chars, 
tokenization, padding/truncation, missing value handling.
"""
import re
import os
import csv
import json
import argparse
import numpy as np
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
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

def tokenize_data(data, tokenizer, max_length):
    """Tokenize a list of conversation pairs"""
    input_ids_list = []
    attn_list = []
    labels_list = []
    
    for pair in tqdm(data, desc="Tokenizing"):
        prompt = f"User: {pair['user_input']} Bot:"
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
    
    return {
        "input_ids": np.stack(input_ids_list),
        "attention_mask": np.stack(attn_list),
        "labels": np.stack(labels_list)
    }

def main(raw_csv="data/cinema_tmdb_dataset.csv", 
         clean_json="data/cinema_tmdb_clean.json",
         train_save="data/tokenized_train.npz",
         val_save="data/tokenized_val.npz",
         model_name="gpt2", 
         max_length=128,
         val_split=0.1):
    
    os.makedirs("data", exist_ok=True)
    
    # Load and clean data
    print("Loading and cleaning data...")
    raw = load_csv(raw_csv)
    cleaned = []
    
    for r in raw:
        user = clean_text(r.get("user_input",""))
        bot = clean_text(r.get("bot_response",""))
        if user == "" or bot == "":
            # Handling missing values: skip (documented)
            continue
        cleaned.append({
            "id": r.get("id",""), 
            "user_input": user, 
            "bot_response": bot
        })
    
    print(f"Cleaned {len(cleaned)} conversation pairs")
    
    # Save cleaned JSON
    with open(clean_json, "w", encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    
    # Split into train/validation
    print(f"Splitting data: {int((1-val_split)*100)}% train, {int(val_split*100)}% validation")
    train_data, val_data = train_test_split(
        cleaned, 
        test_size=val_split, 
        random_state=42,
        shuffle=True
    )
    
    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    
    # Tokenize training data
    print("\nTokenizing training data...")
    train_arrays = tokenize_data(train_data, tokenizer, max_length)
    np.savez_compressed(train_save, **train_arrays)
    print(f"Saved training dataset to: {train_save}")
    print(f"Training dataset size: {train_arrays['input_ids'].shape[0]}")
    
    # Tokenize validation data
    print("\nTokenizing validation data...")
    val_arrays = tokenize_data(val_data, tokenizer, max_length)
    np.savez_compressed(val_save, **val_arrays)
    print(f"Saved validation dataset to: {val_save}")
    print(f"Validation dataset size: {val_arrays['input_ids'].shape[0]}")
    
    print("\n✅ Preprocessing complete!")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val:   {len(val_data)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", default="data/cinema_tmdb_dataset.csv")
    parser.add_argument("--clean_json", default="data/cinema_tmdb_clean.json")
    parser.add_argument("--train_save", default="data/tokenized_train.npz")
    parser.add_argument("--val_save", default="data/tokenized_val.npz")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--val_split", type=float, default=0.1, 
                        help="Validation split ratio (default: 0.1 = 10%)")
    args = parser.parse_args()
    
    main(raw_csv=args.raw_csv, clean_json=args.clean_json, 
         train_save=args.train_save, val_save=args.val_save,
         max_length=args.max_length, model_name=args.model_name,
         val_split=args.val_split)