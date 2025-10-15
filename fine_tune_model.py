import os
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM, create_optimizer
from datasets import Dataset
import numpy as np

MODEL_NAME = "distilgpt2"
DATASET_PATH = "data/movie_dialogue_pairs.csv"
OUTPUT_DIR = "./results_movie_chatbot"
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
MAX_LENGTH = 128

def fine_tune_chatbot_tensorflow():
    """
    Fine-tune a chatbot using TensorFlow and Hugging Face Transformers
    """
    print(f"Loading dataset from '{DATASET_PATH}'...")
    
    # Load and sample dataset
    n_samples = 40000
    df = pd.read_csv(DATASET_PATH).sample(n=n_samples, random_state=42)
    
    # Format dialogue pairs
    def format_dialogue(row):
        return f"<s>{row['prompt']}</s>{row['response']}</s>"
    
    df['text'] = df.apply(format_dialogue, axis=1)
    print(f"Dataset loaded and formatted with {len(df)} samples.")
    
    # Load tokenizer
    print(f"Loading tokenizer for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize function
    def tokenize_function(examples):
        result = tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            return_tensors='np'
        )
        # For causal LM, labels are the same as input_ids
        result['labels'] = result['input_ids'].copy()
        return result
    
    print("Tokenizing dataset...")
    hg_dataset = Dataset.from_pandas(df[['text']])
    tokenized_dataset = hg_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Convert to TensorFlow dataset
    print("Converting to TensorFlow dataset...")
    tf_dataset = tokenized_dataset.to_tf_dataset(
        columns=['input_ids', 'attention_mask', 'labels'],
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: x
    )
    
    # Load TensorFlow model
    print(f"Loading TensorFlow model '{MODEL_NAME}'...")
    model = TFAutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Calculate training steps
    num_train_steps = (len(df) // BATCH_SIZE) * NUM_TRAIN_EPOCHS
    
    # Create optimizer with learning rate schedule
    optimizer, lr_schedule = create_optimizer(
        init_lr=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_train_steps // 10,
        weight_decay_rate=0.01
    )
    
    # Compile model
    print("Compiling model...")
    model.compile(optimizer=optimizer)
    
    # Training
    print("\nStarting TensorFlow model fine-tuning...")
    print(f"Training for {NUM_TRAIN_EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print("-" * 50)
    
    history = model.fit(
        tf_dataset,
        epochs=NUM_TRAIN_EPOCHS,
        verbose=1
    )
    
    print("\nFine-tuning complete!")
    
    # Save model in TensorFlow format
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    
    print(f"\nSaving model to '{final_model_path}'...")
    
    # Save using TensorFlow SavedModel format
    model.save_pretrained(final_model_path, saved_model=True)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"✓ Model saved successfully in TensorFlow format!")
    print(f"✓ Tokenizer saved successfully!")
    
    # Print training summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Model: {MODEL_NAME}")
    print(f"Training samples: {len(df)}")
    print(f"Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Model saved to: {final_model_path}")
    print("=" * 50)
    
    return model, tokenizer, history

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at '{DATASET_PATH}'")
        print("Please run 'prepare_dataset.py' first to create the dataset.")
        exit(1)
    
    # Run fine-tuning
    model, tokenizer, history = fine_tune_chatbot_tensorflow()
    
    # Test the model
    print("\n" + "=" * 50)
    print("TESTING THE FINE-TUNED MODEL")
    print("=" * 50)
    
    test_prompts = [
        "What's your favorite movie?",
        "Can you recommend a good film?",
        "Tell me about action movies"
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(f"<s>{prompt}</s>", return_tensors="tf")
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")