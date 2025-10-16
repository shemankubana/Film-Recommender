# evaluate/evaluate.py

import argparse
import json
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Ensure you have all necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab') # <-- CHECK FOR THE NEW RESOURCE
except LookupError:
    print("NLTK resources not found. Downloading...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True) # <-- DOWNLOAD THE NEW RESOURCE

def generate_response(model, tokenizer, prompt, max_length=100):
    """Generates a response from the model given a prompt."""
    # The tokenizer now returns a dictionary with input_ids and attention_mask
    encoding = tokenizer(prompt, return_tensors='pt').to(model.device)

    # We pass both input_ids and attention_mask to the model
    outputs = model.generate(
        **encoding, # <-- PASS THE ENTIRE ENCODING DICTIONARY
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2
        # 'early_stopping' was removed as it's not a valid flag here
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

def evaluate_model(model, tokenizer, data, n_eval):
    """Evaluates the model on a subset of the data and calculates BLEU and ROUGE scores."""
    results = []
    eval_data = data[:n_eval] if n_eval else data

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    for item in tqdm(eval_data, desc="Evaluating"):
        prompt = item['user_input']
        reference_response = item['bot_response']
        
        generated_response = generate_response(model, tokenizer, prompt)
        
        reference_tokens = [nltk.word_tokenize(reference_response)]
        generated_tokens = nltk.word_tokenize(generated_response)

        from nltk.translate.bleu_score import SmoothingFunction
        
        # ... inside the evaluate_model function ...
        chencherry = SmoothingFunction()
        bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=chencherry.method1)

        rouge_scores = scorer.score(reference_response, generated_response)
        
        results.append({
            'prompt': prompt,
            'reference_response': reference_response,
            'generated_response': generated_response,
            'bleu_score': bleu_score,
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure
        })
        
    return pd.DataFrame(results)

def main(model_dir, clean_json, n_eval):
    """Main function to load the model and run evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {model_dir}")
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"Loading data from: {clean_json}")
    with open(clean_json, 'r') as f:
        data = json.load(f)

    evaluation_df = evaluate_model(model, tokenizer, data, n_eval)

    print("\nEvaluation Results Summary:")
    print(evaluation_df[['bleu_score', 'rouge1', 'rougeL']].describe())

    output_path = "evaluation_results.csv"
    evaluation_df.to_csv(output_path, index=False)
    print(f"\nFull evaluation results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned GPT-2 model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the fine-tuned model is saved.")
    parser.add_argument("--clean_json", type=str, required=True, help="Path to the clean JSON data for evaluation.")
    parser.add_argument("--n_eval", type=int, default=None, help="Number of samples to evaluate on. Default is all.")
    
    args = parser.parse_args()
    main(model_dir=args.model_dir, clean_json=args.clean_json, n_eval=args.n_eval)