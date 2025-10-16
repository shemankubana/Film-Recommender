import re
import os
import json
import argparse
import numpy as np
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|endoftext|>"

def tokenize_data(data, tokenizer, max_length):
    """Tokenize a list of conversation pairs"""
    input_ids_list = []
    attn_list = []
    labels_list = []
    
    for pair in tqdm(data, desc="Tokenizing"):
        prompt = f"User: {pair['user_input']} Bot:"
        target = f" {pair['bot_response']}{EOS_TOKEN}"
        full = prompt + target
        
        enc_full = tokenizer(full, truncation=True, max_length=max_length, padding='max_length')
        input_ids = np.array(enc_full["input_ids"], dtype=np.int32)
        attn = np.array(enc_full["attention_mask"], dtype=np.int32)
        
        enc_prompt = tokenizer(prompt, truncation=True, max_length=max_length)
        prompt_len = len(enc_prompt["input_ids"])
        
        labels = input_ids.copy()
        labels[:prompt_len] = -100
        
        input_ids_list.append(input_ids)
        attn_list.append(attn)
        labels_list.append(labels)
    
    return {
        "input_ids": np.stack(input_ids_list),
        "attention_mask": np.stack(attn_list),
        "labels": np.stack(labels_list)
    }

def main(input_json, 
         train_save="data/tokenized_train.npz",
         val_save="data/tokenized_val.npz",
         model_name="gpt2", 
         max_length=128,
         val_split=0.1):
    
    os.makedirs("data", exist_ok=True)
    
    print(f"Loading data from {input_json}...")
    with open(input_json, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} conversation pairs")
    
    train_data, val_data = train_test_split(
        data, test_size=val_split, random_state=42, shuffle=True
    )
    
    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Add pad token and also tell the model about our EOS token
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN, "eos_token": EOS_TOKEN})
    
    print("\nTokenizing training data...")
    train_arrays = tokenize_data(train_data, tokenizer, max_length)
    np.savez_compressed(train_save, **train_arrays)
    print(f"Saved training dataset to: {train_save}")
    
    print("\nTokenizing validation data...")
    val_arrays = tokenize_data(val_data, tokenizer, max_length)
    np.savez_compressed(val_save, **val_arrays)
    print(f"Saved validation dataset to: {val_save}")
    
    print("\n✅ Preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and tokenize conversational data.")
    
    # Argument for the input JSON file
    parser.add_argument("--input_json", default="data/cinema_tmdb_augmented.json", help="Path to the input JSON file.")
    
    # Arguments for the output files
    parser.add_argument("--train_save", default="data/tokenized_train.npz", help="Path to save the tokenized training data.")
    parser.add_argument("--val_save", default="data/tokenized_val.npz", help="Path to save the tokenized validation data.")
    
    # Arguments for tokenization and model settings
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length for tokenizer.")
    parser.add_argument("--model_name", default="gpt2", help="Name of the pre-trained model for the tokenizer.")
    
    # Argument for the train/validation split
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio (default: 0.1 = 10%).")
    
    args = parser.parse_args()
    
    # Pass all arguments to the main function
    main(
        input_json=args.input_json, 
        train_save=args.train_save, 
        val_save=args.val_save,
        max_length=args.max_length, 
        model_name=args.model_name,
        val_split=args.val_split
    )