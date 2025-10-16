from huggingface_hub import Repository

# Path to your fine-tuned model
model_folder = "saved_model/gpt2_cinema_tf"

# Your Hugging Face repo ID (already created)
repo_id = "nkubana/gpt2-cinema"

# Push the folder to Hugging Face Hub
repo = Repository(local_dir=model_folder, clone_from=repo_id)
repo.push_to_hub(commit_message="Update fine-tuned GPT-2 model")
