import json

def filter_for_qa(input_file, output_file):
    """
    Filters the augmented dataset to only include factual Q&A pairs.
    """
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    qa_data = []
    qa_keywords = ['directed', 'starred in', 'released']

    for entry in data:
        prompt = entry['user_input'].lower()
        # Check if any of the keywords are in the prompt
        if any(keyword in prompt for keyword in qa_keywords):
            qa_data.append(entry)

    print(f"Original size: {len(data)}")
    print(f"Filtered Q&A size: {len(qa_data)}")
    
    with open(output_file, 'w') as f:
        json.dump(qa_data, f, indent=2)
        
    print(f"Saved Q&A-only dataset to {output_file}")

if __name__ == "__main__":
    filter_for_qa('data/cinema_tmdb_augmented.json', 'data/cinema_tmdb_qa_only.json')