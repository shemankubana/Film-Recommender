import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TFAutoModelForCausalLM, TFTrainingArguments, TFTrainer

# Configuration
MODEL_NAME = "distilgpt2"
DATASET_PATH = "data/movie_dialogue_pairs.csv"
OUTPUT_DIR = "./results_movie_chatbot"
NUM_TRAIN_EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 5e-5

def fine_tune_chatbot():
    """
    Loads the dataset, tokenizes it, and fine-tunes the distilgpt2 model.
    """
    print(f"Loading dataset from '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)

    def format_dialogue(row):
        return f"<s>{row['prompt']}</s>{row['response']}</s>"

    df['text'] = df.apply(format_dialogue, axis=1)
    
    hg_dataset = Dataset.from_pandas(df[['text']])
    print("Dataset loaded and formatted.")
    
    print(f"Loading tokenizer and model for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, pad_token='<|endoftext|>')
    model = TFAutoModelForCausalLM.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128)

    print("Tokenizing the dataset... (This might take a while)")
    tokenized_dataset = hg_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print("Tokenization complete.")

    training_args = TFTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        push_to_hub=False,
    )

    trainer = TFTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("\nStarting model fine-tuning...")
    trainer.train()
    print("Fine-tuning complete!")

    #Save the Final Model ---
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    print(f"Model saved to '{final_model_path}'")


if __name__ == "__main__":
    fine_tune_chatbot()