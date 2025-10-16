"""
Train / fine-tune GPT-2 (PyTorch) on the tokenized dataset WITH VALIDATION.
Saves model to --model_dir and logs experiment to experiment_results.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datetime import datetime
from tqdm import tqdm

EXPERIMENT_LOG = "experiment_results.csv"

def load_dataset(npz_path, batch_size=8, shuffle=True):
    """Load tokenized dataset from npz file"""
    data = np.load(npz_path)
    input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(data["attention_mask"], dtype=torch.long)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader, len(dataset)

def compute_perplexity(loss):
    """Compute perplexity from loss"""
    return float(np.exp(loss)) if loss < 100 else float("inf")

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for batch in progress_bar:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)

def main(train_path="data/tokenized_train.npz",
         val_path="data/tokenized_val.npz", 
         model_name="gpt2", 
         model_dir="saved_model/gpt2_cinema",
         batch_size=8, 
         lr=5e-5, 
         epochs=3,
         experiment_name="baseline",
         warmup_steps=100):
    
    # Create output directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # Load datasets
    train_loader, train_size = load_dataset(train_path, batch_size=batch_size, shuffle=True)
    val_loader, val_size = load_dataset(val_path, batch_size=batch_size, shuffle=False)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Print experiment info
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")
    print(f"Training dataset size: {train_size}")
    print(f"Validation dataset size: {val_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Total training steps: {total_steps}")
    print(f"{'='*60}\n")
    
    best_val_loss = float("inf")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        train_perplexity = compute_perplexity(train_loss)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        val_perplexity = compute_perplexity(val_loss)
        
        # Print results
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Perplexity: {train_perplexity:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Perplexity: {val_perplexity:.4f}")
        print(f"{'─'*60}\n")
        
        # Save checkpoint
        ckpt_dir = os.path.join(model_dir, f"epoch_{epoch}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"✅ Checkpoint saved to {ckpt_dir}")
        
        # Log to CSV
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_name": experiment_name,
            "model_dir": ckpt_dir,
            "batch_size": batch_size,
            "lr": lr,
            "epochs_trained": epoch,
            "train_loss": float(train_loss),
            "train_perplexity": float(train_perplexity),
            "val_loss": float(val_loss),
            "val_perplexity": float(val_perplexity)
        }
        
        df = pd.DataFrame([row])
        header = not os.path.exists(EXPERIMENT_LOG)
        df.to_csv(EXPERIMENT_LOG, mode="a", index=False, header=header)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = os.path.join(model_dir, "best")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"✅ New best model saved! Val Loss: {best_val_loss:.4f}\n")
    
    print(f"\n{'='*60}")
    print("Training finished!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {compute_perplexity(best_val_loss):.4f}")
    print(f"Models saved under: {model_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/tokenized_train.npz")
    parser.add_argument("--val_path", default="data/tokenized_val.npz")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--model_dir", default="saved_model/gpt2_cinema")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--experiment_name", default="baseline")
    parser.add_argument("--warmup_steps", type=int, default=100)
    args = parser.parse_args()
    
    main(
        train_path=args.train_path,
        val_path=args.val_path,
        model_name=args.model_name, 
        model_dir=args.model_dir,
        batch_size=args.batch_size, 
        lr=args.lr, 
        epochs=args.epochs,
        experiment_name=args.experiment_name,
        warmup_steps=args.warmup_steps
    )