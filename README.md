# Film Recommender

This project features a chatbot fine-tuned from **GPT-2** to answer factual questions about movies — such as directors, cast, and release dates. It was developed as part of a **summative assignment**.

## Features

* **Factual Q&A:** Specialized to answer movie-related questions accurately.
* **Streamlit UI:** Interactive and user-friendly web interface for chatting with the model.
* **Data Augmentation:** Enhanced dataset diversity for better generalization.
* **Early Stopping:** Prevents overfitting by monitoring validation loss.
* **Rejects Out-of-Domain Queries:** Politely refuses non-factual questions (e.g., recommendations).

## Project Structure

```
/
├── data/                 # Raw, processed, and tokenized datasets
├── evaluate/             # Model evaluation scripts
├── saved_model/          # Saved model checkpoints
├── app.py                # Streamlit web application
├── train.py              # Model training script
├── preprocess.py         # Data preprocessing and tokenization
├── augment_data.py       # Dataset augmentation script
├── create_qa_dataset.py  # Extracts and filters Q&A pairs
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Setup and Installation

Follow the steps below to set up and run the project locally:

1. **Clone the Repository**

   ```bash
   git clone [https://github.com/nkubana0/Film-Recommender.git](url)
   cd Film-Recommender
   ```

2. **Create a Conda Environment and Install Dependencies**

   ```bash
   # Recommended: Python 3.12
   conda create --name cinemaqa python=3.12
   conda activate cinemaqa
   pip install -r requirements.txt
   ```

## Usage Guide

Run the following steps sequentially to train and use the chatbot:

1. **Data Augmentation (Optional but Recommended)**

   ```bash
   python augment_data.py
   ```

2. **Filter for Q&A Data**

   ```bash
   python create_qa_dataset.py
   ```

3. **Preprocess and Tokenize**

   ```bash
   python preprocess.py --input_json data/cinema_tmdb_qa_only.json
   ```

4. **Train the Model**

   ```bash
   python train.py --experiment_name "final_model" --epochs 10 --patience 2
   ```

5. **Run the Chatbot Application**

   ```bash
   streamlit run app.py
   ```

## Model Performance

The final model, **`exp5_qa_only_focused`**, was trained on a curated Q&A dataset and achieved:

* **Mean ROUGE-1 Score:** `0.31` on the held-out test set
  (indicating strong factual recall and overlap with ground truth)

It performs well on factual questions while correctly **ignoring subjective or out-of-domain prompts** by responding with a blank or neutral message.

---
