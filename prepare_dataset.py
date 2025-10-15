import os
import pandas as pd
from tqdm import tqdm

LOCAL_DATA_PATH = "archive/" 

def prepare_cornell_data_from_local():
    """
    Processes the Cornell Movie Dialogs Corpus from a local directory
    into a CSV of prompt-response pairs.
    """
    output_dir = "data"
    output_csv = os.path.join(output_dir, "movie_dialogue_pairs.csv")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # A simple check to ensure the path has been updated
    if not os.path.isdir(LOCAL_DATA_PATH) or "path/to/your/archive" in LOCAL_DATA_PATH:
        print("="*60)
        print("!!! ACTION REQUIRED !!!")
        print("Please update the 'LOCAL_DATA_PATH' variable in this script")
        print("to the full path of your 'archive' folder.")
        print("="*60)
        return None

    # Check if the final file already exists
    if os.path.exists(output_csv):
        print(f"'{output_csv}' already exists. Skipping processing.")
        return pd.read_csv(output_csv)

    lines_file = os.path.join(LOCAL_DATA_PATH, "movie_lines.txt")
    convos_file = os.path.join(LOCAL_DATA_PATH, "movie_conversations.txt")

    # Check if the source files can be found
    if not os.path.exists(lines_file) or not os.path.exists(convos_file):
        print(f"ERROR: Could not find 'movie_lines.txt' or 'movie_conversations.txt' in the specified path:")
        print(f"'{LOCAL_DATA_PATH}'")
        return None

    # --- 2. Load Lines into a Dictionary ---
    print("Loading movie lines...")
    lines = {}
    with open(lines_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 5:
                line_id, _, _, _, line_text = parts
                lines[line_id] = line_text

    # --- 3. Create Prompt-Response Pairs ---
    print("Creating prompt-response pairs...")
    prompts = []
    responses = []
    with open(convos_file, 'r', encoding='iso-8859-1') as f:
        # Use tqdm for a progress bar, as this can take a moment
        for line in tqdm(f, desc="Processing conversations"):
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 4:
                convo_ids = eval(parts[3])
                for i in range(len(convo_ids) - 1):
                    prompt_id = convo_ids[i]
                    response_id = convo_ids[i + 1]
                    if prompt_id in lines and response_id in lines:
                        prompts.append(lines[prompt_id])
                        responses.append(lines[response_id])

    # --- 4. Save to a DataFrame and CSV ---
    print(f"Created {len(prompts)} pairs.")
    df = pd.DataFrame({"prompt": prompts, "response": responses})
    
    df = df.dropna()
    df = df[df['prompt'].str.len() > 1]
    df = df[df['response'].str.len() > 1]
    
    print(f"Saving {len(df)} cleaned pairs to '{output_csv}'...")
    df.to_csv(output_csv, index=False)
    
    print("Done!")
    return df

if __name__ == "__main__":
    dialogue_df = prepare_cornell_data_from_local()
    if dialogue_df is not None:
        print("\n--- Sample of the processed data ---")
        print(dialogue_df.head())
        print(f"\nTotal pairs: {len(dialogue_df)}")