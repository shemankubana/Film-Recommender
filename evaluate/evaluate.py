"""
Evaluation script: loads saved model and evaluates on the cleaned dataset.
Computes:
- Perplexity (approx from average loss)
- BLEU (nltk) for generated responses
- Token-overlap F1 (simple token-level F1)
Also runs 10 qualitative queries (taken from cleaned dataset) and 3 out-of-domain queries for rejection check.
"""

import os
import argparse
import json
import numpy as np
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import f1_score
from tqdm import tqdm

nltk.download('punkt', quiet=True)

def load_cleaned(clean_json="data/cinema_tmdb_clean.json"):
    with open(clean_json, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_response(model, tokenizer, text, max_length=80):
    # Use model.generate via TF wrapper
    input_ids = tokenizer.encode(text, return_tensors="tf")
    # generate
    gen = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    # we expect continuation; strip the prompt portion
    if text in out:
        out = out.split(text, 1)[1].strip()
    return out

def token_f1(pred, ref):
    # simple token-level F1: treat tokens as set for precision/recall
    pred_tokens = nltk.word_tokenize(pred.lower())
    ref_tokens = nltk.word_tokenize(ref.lower())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(ref_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def main(model_dir="saved_model/gpt2_cinema/best", clean_json="data/cinema_tmdb_clean.json", n_eval=100):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = TFGPT2LMHeadModel.from_pretrained(model_dir)

    data = load_cleaned(clean_json)
    # sample up to n_eval pairs for testing
    eval_pairs = data[:n_eval]

    bleu_scores = []
    f1_scores = []
    # approximate perplexity by computing loss over tokens in eval pairs
    losses = []

    for pair in tqdm(eval_pairs, desc="Evaluating"):
        prompt = f"User: {pair['user_input']}  Bot:"
        # full text to compute loss (labels masked for prompt like during training)
        full = prompt + " " + pair['bot_response']
        enc = tokenizer(full, return_tensors="tf", truncation=True, max_length=128, padding="max_length")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # compute loss by calling model with labels
        # need to prepare labels like in training: mask prompt tokens
        enc_prompt = tokenizer(prompt, return_tensors="tf", truncation=True, max_length=128, padding="max_length")
        prompt_len = tf.reduce_sum(tf.cast(enc_prompt["attention_mask"][0], tf.int32)).numpy()
        labels = input_ids.numpy().copy()
        labels[0, :prompt_len] = -100
        out = model(input_ids, labels=tf.constant(labels))
        loss = float(out.loss.numpy())
        losses.append(loss)

        # generate response
        gen = generate_response(model, tokenizer, prompt, max_length=128)
        # compute BLEU (reference tokens)
        ref_tokens = nltk.word_tokenize(pair['bot_response'].lower())
        hyp_tokens = nltk.word_tokenize(gen.lower())
        smoothie = SmoothingFunction().method4
        try:
            bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        except Exception:
            bleu = 0.0
        bleu_scores.append(bleu)
        f1_scores.append(token_f1(gen, pair['bot_response']))

    avg_loss = float(np.mean(losses))
    perplexity = float(np.exp(avg_loss)) if avg_loss < 100 else float("inf")
    avg_bleu = float(np.mean(bleu_scores))
    avg_f1 = float(np.mean(f1_scores))

    print("Evaluation results:")
    print(f"Avg loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Avg BLEU: {avg_bleu:.4f}")
    print(f"Avg token-F1: {avg_f1:.4f}")

    # qualitative tests (10 in-domain, 3 OOD)
    print("\nQualitative examples:")
    for i, pair in enumerate(eval_pairs[:10]):
        prompt = f"User: {pair['user_input']}  Bot:"
        gen = generate_response(model, tokenizer, prompt, max_length=80)
        print(f"Q: {pair['user_input']}")
        print(f"GT: {pair['bot_response']}")
        print(f"GEN: {gen}\n")

    print("Out-of-domain test (should reject or indicate out-of-scope):")
    ood = ["What's the weather today in Kigali?", "How do I jailbreak my phone?", "Give me the bitcoin price now"]
    for q in ood:
        prompt = f"User: {q}  Bot:"
        gen = generate_response(model, tokenizer, prompt, max_length=80)
        print(f"Q: {q}")
        print(f"GEN: {gen}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="saved_model/gpt2_cinema/best")
    parser.add_argument("--clean_json", default="data/cinema_tmdb_clean.json")
    parser.add_argument("--n_eval", type=int, default=100)
    args = parser.parse_args()
    main(model_dir=args.model_dir, clean_json=args.clean_json, n_eval=args.n_eval)
