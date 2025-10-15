import os
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

MODEL_NAME = "distilgpt2"
DATASET_PATH = "data/movie_dialogue_pairs.csv"
OUTPUT_DIR = "./results_movie_chatbot"
NUM_TRAIN_EPOCHS = 1
BATCH_SIZE = 8 
LEARNING_RATE = 5e-5

def fine_tune_chatbot():
    print(f"Loading dataset from '{DATASET_PATH}'...")
    n_samples = 40000 
    df = pd.read_csv(DATASET_PATH).sample(n=n_samples, random_state=42)
    
    def format_dialogue(row):
        return f"<s>{row['prompt']}</s>{row['response']}</s>"

    df['text'] = df.apply(format_dialogue, axis=1)
    hg_dataset = Dataset.from_pandas(df[['text']])
    print(f"Dataset loaded and formatted with {len(df)} samples.")
    
    print(f"Loading tokenizer and model for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, pad_token='<|endoftext|>')
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128)

    print("Tokenizing the dataset...")
    tokenized_dataset = hg_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print("Tokenization complete.")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="epoch",
        fp16=True, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting model fine-tuning with PyTorch Trainer...")
    trainer.train()
    print("Fine-tuning complete!")

    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    print(f"Model and tokenizer saved to '{final_model_path}'")

if __name__ == "__main__":
    fine_tune_chatbot()