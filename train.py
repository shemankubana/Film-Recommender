"""
Train / fine-tune GPT-2 (TensorFlow) on the tokenized dataset.
Saves model to --model_dir and logs experiment to experiment_results.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm

EXPERIMENT_LOG = "experiment_results.csv"

def make_tf_dataset(npz_path, batch_size=8, shuffle=True):
    data = np.load(npz_path)
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    labels = data["labels"]

    ds = tf.data.Dataset.from_tensor_slices((
        {"input_ids": input_ids, "attention_mask": attention_mask},
        labels
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=2048)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, input_ids.shape[0]

def compute_perplexity(loss):
    return float(np.exp(loss)) if loss < 100 else float("inf")

def main(tokenized="data/tokenized_dataset.npz", model_name="gpt2", model_dir="saved_model/gpt2_cinema",
         batch_size=8, lr=5e-5, epochs=3):
    os.makedirs(model_dir, exist_ok=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = TFGPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    train_ds, n = make_tf_dataset(tokenized, batch_size=batch_size)

    # optimizer — for simplicity using Adam. To use AdamW you may import tfa or transformers optimizer utilities.
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Loss: model returns losses when labels passed; but compile needs a loss – use sparse_categorical_crossentropy masked
    # We'll perform custom training loop to properly pass labels with -100 masks.
    train_loss = tf.keras.metrics.Mean(name="train_loss")

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            outputs = model(inputs, labels=labels, training=True)
            loss = outputs.loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss)
        return loss

    print(f"Starting training: dataset size={n}, batch_size={batch_size}, epochs={epochs}")
    steps_per_epoch = int(np.ceil(n / batch_size))
    best_loss = float("inf")

    for epoch in range(1, epochs+1):
        train_loss.reset_states()
        prog = tqdm(enumerate(train_ds, start=1), total=steps_per_epoch, desc=f"Epoch {epoch}/{epochs}")
        for step, (inputs, labels) in prog:
            loss = train_step(inputs, labels)
            prog.set_postfix({"loss": float(loss)})
        epoch_loss = train_loss.result().numpy()
        perplexity = compute_perplexity(epoch_loss)
        print(f"Epoch {epoch} finished. Loss={epoch_loss:.4f} Perplexity={perplexity:.4f}")

        # save checkpoint each epoch
        ckpt_dir = os.path.join(model_dir, f"epoch_{epoch}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        # log experiment
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_dir": ckpt_dir,
            "batch_size": batch_size,
            "lr": lr,
            "epochs_trained": epoch,
            "loss": float(epoch_loss),
            "perplexity": float(perplexity)
        }
        # append to CSV
        df = pd.DataFrame([row])
        header = not os.path.exists(EXPERIMENT_LOG)
        df.to_csv(EXPERIMENT_LOG, mode="a", index=False, header=header)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # save best model
            best_dir = os.path.join(model_dir, "best")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    print("Training finished. Best loss:", best_loss)
    print("Models saved under:", model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized", default="data/tokenized_dataset.npz")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--model_dir", default="saved_model/gpt2_cinema")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(tokenized=args.tokenized, model_name=args.model_name, model_dir=args.model_dir,
         batch_size=args.batch_size, lr=args.lr, epochs=args.epochs)
