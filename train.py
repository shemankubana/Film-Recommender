import argparse
import json
import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW # <-- THIS IS THE KEY CHANGE
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_scheduler # <-- AdamW removed from here
from tqdm import tqdm
from datetime import datetime

# HELPER FUNCTIONS

def load_tokenized_data(path):
    """Loads a .npz file into a TensorDataset."""
    data = np.load(path)
    dataset = TensorDataset(
        torch.tensor(data["input_ids"], dtype=torch.long),
        torch.tensor(data["attention_mask"], dtype=torch.long),
        torch.tensor(data["labels"], dtype=torch.long),
    )
    return dataset

# TRAINING AND EVALUATION LOGIC

def main(
    experiment_name,
    train_path,
    val_path,
    model_name="gpt2",
    epochs=3,
    batch_size=8,
    lr=5e-5,
    patience=2
):
    
    # SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = "saved_model/gpt2_cinema"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>", "eos_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    # DATA LOADING
    train_dataset = load_tokenized_data(train_path)
    val_dataset = load_tokenized_data(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # RAINING SETUP
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    print("\n" + "="*60)
    print(f"Experiment: {experiment_name}")
    print("="*60)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Early Stopping Patience: {patience}")
    print(f"Total training steps: {num_training_steps}")
    print("="*60 + "\n")

    # EARLY STOPPING INITIALIZATION
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # TRAINING LOOP
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 60)
        
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch in train_progress_bar:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc="Evaluating", leave=False)
            for batch in val_progress_bar:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()
                val_progress_bar.set_postfix({'val_loss': loss.item()})

        avg_val_loss = total_val_loss / len(val_loader)
        
        train_perplexity = math.exp(avg_train_loss)
        val_perplexity = math.exp(avg_val_loss)

        print("\n" + "─"*60)
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Perplexity: {train_perplexity:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Perplexity: {val_perplexity:.4f}")
        print("─"*60)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            best_model_path = os.path.join(output_dir, "best")
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"\n✅ New best model saved! Val Loss: {avg_val_loss:.4f}\n")
        else:
            epochs_no_improve += 1
            print(f"\nValidation loss did not improve for {epochs_no_improve} epoch(s).")
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break

    print("\n" + "="*60)
    print("Training finished!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved under: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--train_path", default="data/tokenized_train.npz")
    parser.add_argument("--val_path", default="data/tokenized_val.npz")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=2, help="How many epochs to wait for validation loss improvement before stopping.")
    
    args = parser.parse_args()
    main(**vars(args))