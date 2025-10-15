# refresh_tokenizer.py

from transformers import AutoTokenizer
import os

BASE_MODEL_NAME = "distilgpt2"
FINAL_MODEL_PATH = "./results_movie_chatbot/final_model"

def refresh_tokenizer_files():
    """
    Downloads a fresh copy of the base tokenizer and saves it
    to the final model directory, overwriting any corrupted files.
    """
    print(f"Downloading fresh tokenizer for '{BASE_MODEL_NAME}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        
        print(f"Saving fresh tokenizer to '{FINAL_MODEL_PATH}'...")
        tokenizer.save_pretrained(FINAL_MODEL_PATH)
        
        print("\n✅ Tokenizer files have been successfully refreshed!")
        print("You can now run your Streamlit app again.")
    
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your internet connection and the model paths.")

if __name__ == "__main__":
    if not os.path.isdir(FINAL_MODEL_PATH):
        print(f"Error: The directory '{FINAL_MODEL_PATH}' does not exist.")
        print("Please make sure you have run the fine-tuning script first.")
    else:
        refresh_tokenizer_files()