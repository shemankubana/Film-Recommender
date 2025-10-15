# Cinema Domain Chatbot (GPT-2, TensorFlow) — Summative Assignment

Overview
--------
This project builds a cinema-domain conversational chatbot using a pre-trained GPT-2 model fine-tuned with TensorFlow. Data is collected from TMDB API and converted into conversational Q&A pairs. The project implements preprocessing, model fine-tuning, evaluation (perplexity, BLEU, token-F1), and a Gradio UI.

Required files (created by scripts)
- data/cinema_tmdb_dataset.csv   # created by data/collect_tmdb.py
- data/cinema_tmdb_clean.json   # created by preprocess.py
- data/tokenized_dataset.npz    # created by preprocess.py
- saved_model/                   # model checkpoints saved by train.py
- experiment_results.csv         # logs of hyperparameter experiments

Setup
-----
1. Clone/copy project files.
2. Install dependencies:
   python -m pip install -r requirements.txt
3. Get a TMDB API key: https://www.themoviedb.org/settings/api
   Set in environment:
   export TMDB_API_KEY="your_key_here"     (Linux / macOS)
   setx TMDB_API_KEY "your_key_here"      (Windows powershell)

Usage
-----
1. Collect TMDB data (example uses top N popular movies):
   python data/collect_tmdb.py --num_movies 200 --out data/cinema_tmdb_dataset.csv

2. Preprocess & tokenize:
   python preprocess.py --raw_csv data/cinema_tmdb_dataset.csv --max_length 128

3. Train (example):
   python train.py --batch_size 8 --lr 5e-5 --epochs 3 --model_dir saved_model/gpt2_cinema

4. Evaluate:
   python evaluate.py --model_dir saved_model/gpt2_cinema

5. Run UI (loads model from model_dir):
   python webapp/gradio_app.py --model_dir saved_model/gpt2_cinema

What to include in submission PDF
--------------------------------
- Problem definition & domain alignment (purpose).
- Dataset collection method (TMDB), data size, sample rows.
- Preprocessing steps (cleaning regex, tokenization details).
- Model choice & training hyperparameters (table of experiments).
- Evaluation metrics: perplexity, BLEU, F1, qualitative samples (including out-of-domain rejection results).
- UI screenshot and demo video link.

Notes
-----
- The tokenizer uses GPT-2 tokenizer and a pad token is added (GPT-2 does not have pad_token by default).
- Training uses TFGPT2LMHeadModel (transformers) and a standard TF training loop.
- The scripts emphasize reproducibility and logging.

