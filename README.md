```markdown
# CinemaQA Chatbot: A Fine-Tuned GPT-2 Film Recommender

This project contains a chatbot built by fine-tuning the GPT-2 language model to answer factual questions about movies, such as directors, cast, and release dates. The project was developed as a summative assignment.

## Features

-   **Factual Q&A:** The model is specialized to answer questions about movie details.
-   **Streamlit UI:** A simple and interactive web interface to chat with the model.
-   **Data Augmentation:** The training data was augmented to improve model robustness.
-   **Early Stopping:** The training process uses early stopping to prevent overfitting and ensure the best model is saved.
-   **Rejects Out-of-Domain Queries:** The final model correctly ignores questions (like recommendations) that it was not trained on.

## Project Structure

```

/
├── data/                 \# Holds raw, processed, and tokenized datasets
├── evaluate/             \# Scripts for model evaluation
├── saved\_model/          \# Saved model checkpoints
├── app.py                \# The Streamlit web application
├── train.py              \# Script for training the model
├── preprocess.py         \# Script for data preprocessing and tokenization
├── augment\_data.py       \# Script for augmenting the dataset
├── create\_qa\_dataset.py  \# Script to filter for Q\&A pairs
├── requirements.txt      \# Project dependencies
└── README.md             \# This file

````

## Setup and Installation

Follow these steps to set up the environment and run the project.

1.  **Clone the repository:**
    ```bash
    git clone [Link to your GitHub Repository]
    cd Film-Recommender
    ```

2.  **Create a Conda environment and install dependencies:**
    ```bash
    # It is recommended to use Python 3.12
    conda create --name cinemaqa python=3.12
    conda activate cinemaqa
    pip install -r requirements.txt
    ```

## Usage

The project is run in a sequence of steps.

1.  **Data Augmentation (Optional but Recommended):**
    ```bash
    python augment_data.py
    ```

2.  **Filter for Q&A Data:**
    ```bash
    python create_qa_dataset.py
    ```

3.  **Preprocess and Tokenize:**
    ```bash
    python preprocess.py --input_json data/cinema_tmdb_qa_only.json
    ```

4.  **Train the Model:**
    ```bash
    python train.py --experiment_name "final_model" --epochs 10 --patience 2
    ```

5.  **Run the Chatbot Application:**
    ```bash
    streamlit run app.py
    ```

## Final Model Performance

The final model (`exp6_qa_only_focused`) was trained on the specialized Q&A dataset. On a held-out test set, it achieved a **mean ROUGE-1 score of 0.31**, indicating a strong factual recall and significant overlap with the ground-truth answers.

The model shows strong performance on factual questions but correctly rejects subjective recommendations and out-of-domain topics by providing a blank response.
````
