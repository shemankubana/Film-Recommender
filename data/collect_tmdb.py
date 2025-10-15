"""
Collect movie facts from TMDB and convert to conversational pairs.
Outputs: data/cinema_tmdb_dataset.csv
"""

import os
import csv
import time
import argparse
import requests
from tqdm import tqdm

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", None)
BASE_URL = "https://api.themoviedb.org/3"

if not TMDB_API_KEY:
    raise EnvironmentError("Please set TMDB_API_KEY environment variable before running this script.")

def search_movie(title):
    r = requests.get(f"{BASE_URL}/search/movie", params={"api_key": TMDB_API_KEY, "query": title})
    r.raise_for_status()
    data = r.json()
    return data.get("results", [])

def get_movie_details(movie_id):
    r = requests.get(f"{BASE_URL}/movie/{movie_id}", params={"api_key": TMDB_API_KEY})
    r.raise_for_status()
    return r.json()

def get_credits(movie_id):
    r = requests.get(f"{BASE_URL}/movie/{movie_id}/credits", params={"api_key": TMDB_API_KEY})
    r.raise_for_status()
    return r.json()

def get_recommendations(movie_id):
    r = requests.get(f"{BASE_URL}/movie/{movie_id}/recommendations", params={"api_key": TMDB_API_KEY})
    r.raise_for_status()
    return r.json()

def fetch_popular_movies(page=1):
    r = requests.get(f"{BASE_URL}/movie/popular", params={"api_key": TMDB_API_KEY, "page": page})
    r.raise_for_status()
    return r.json()

def build_pairs_for_movie(movie_id, details, credits, recs, max_rec=3):
    title = details.get("title", "")
    year = details.get("release_date","")[:4] if details.get("release_date") else ""
    director = ""
    for c in credits.get("crew", []):
        if c.get("job") == "Director":
            director = c.get("name")
            break
    cast_list = [c.get("name") for c in credits.get("cast", [])][:5]
    rec_titles = [r.get("title") for r in recs.get("results", [])][:max_rec]
    pairs = []
    base_id = movie_id
    pairs.append({"id": f"{base_id}_dir", "user_input": f"Who directed {title}?", "bot_response": f"{title} was directed by {director}."})
    pairs.append({"id": f"{base_id}_cast", "user_input": f"Who starred in {title}?", "bot_response": f"The main cast includes {', '.join(cast_list)}."})
    if year:
        pairs.append({"id": f"{base_id}_year", "user_input": f"When was {title} released?", "bot_response": f"{title} was released in {year}."})
    if rec_titles:
        pairs.append({"id": f"{base_id}_rec", "user_input": f"Recommend a movie like {title}.", "bot_response": f"If you liked {title}, you might enjoy {', '.join(rec_titles)}."})
    return pairs

def main(out_csv="data/cinema_tmdb_dataset.csv", num_movies=200):
    os.makedirs("data", exist_ok=True)
    pairs = []
    page = 1
    # fetch popular movies pages until we have ~num_movies
    print("Collecting popular movies from TMDB...")
    while len(pairs) < num_movies:
        data = fetch_popular_movies(page=page)
        results = data.get("results", [])
        if not results:
            break
        for item in results:
            try:
                movie_id = item["id"]
                details = get_movie_details(movie_id)
                credits = get_credits(movie_id)
                recs = get_recommendations(movie_id)
                movie_pairs = build_pairs_for_movie(movie_id, details, credits, recs)
                pairs.extend(movie_pairs)
                if len(pairs) >= num_movies:
                    break
                time.sleep(0.25)  # be polite with API
            except Exception as e:
                print("Skipping movie due to error:", e)
        page += 1
        if page > 50:
            break

    # deduplicate by user_input just in case
    seen = set()
    unique = []
    for p in pairs:
        if p["user_input"] not in seen:
            unique.append(p)
            seen.add(p["user_input"])

    # write to CSV
    print(f"Saving {len(unique)} conversational pairs to {out_csv}")
    with open(out_csv, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id","user_input","bot_response"])
        writer.writeheader()
        for row in unique:
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cinema_tmdb_dataset.csv")
    parser.add_argument("--num_movies", type=int, default=200, help="Approximate number of movie records to process (more movies -> more pairs)")
    args = parser.parse_args()
    main(out_csv=args.out, num_movies=args.num_movies)
