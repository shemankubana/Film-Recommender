import json
import random

def augment_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    augmented_data = []
    
    # Templates for augmenting questions
    director_templates = [
        "who directed {movie_title}?",
        "can you tell me the director of {movie_title}?",
        "who was the director for {movie_title}?",
        "tell me who directed {movie_title}.",
    ]
    
    cast_templates = [
        "who starred in {movie_title}?",
        "who was in the cast of {movie_title}?",
        "main cast for {movie_title} please.",
        "list the main actors in {movie_title}.",
    ]
    
    release_templates = [
        "when was {movie_title} released?",
        "what was the release date for {movie_title}?",
        "release date of {movie_title}?",
        "{movie_title} release year?",
    ]

    print(f"Original dataset size: {len(data)}")

    for entry in data:
        # We need to infer the movie title from the original prompt.
        # This is a simple heuristic and might need adjustment.
        original_prompt = entry['user_input']
        
        try:
            if 'directed' in original_prompt:
                movie_title = original_prompt.split('directed ')[1].replace('?', '').strip()
                templates = director_templates
            elif 'starred in' in original_prompt:
                movie_title = original_prompt.split('starred in ')[1].replace('?', '').strip()
                templates = cast_templates
            elif 'released' in original_prompt:
                movie_title = original_prompt.split('was ')[1].split(' released')[0].strip()
                templates = release_templates
            else:
                # If it's a recommendation or other type, just keep the original
                augmented_data.append(entry)
                continue

            # Create new entries from templates
            for template in templates:
                new_entry = {
                    'id': f"{entry['id']}_{random.randint(1000, 9999)}",
                    'user_input': template.format(movie_title=movie_title),
                    'bot_response': entry['bot_response']
                }
                augmented_data.append(new_entry)

        except IndexError:
            # If our simple title parsing fails, just keep the original entry
            augmented_data.append(entry)
            continue
            
    # Shuffle the data to ensure randomness
    random.shuffle(augmented_data)

    print(f"Augmented dataset size: {len(augmented_data)}")
    
    with open(output_file, 'w') as f:
        json.dump(augmented_data, f, indent=2)
        
    print(f"Saved augmented data to {output_file}")


if __name__ == "__main__":
    augment_data('data/cinema_tmdb_clean.json', 'data/cinema_tmdb_augmented.json')