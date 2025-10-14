# utils.py
import os
import zipfile
import requests
import io
import pandas as pd
import re
from pathlib import Path
import streamlit as st

MOVIELENS_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path("data")
MOVIES_CSV = DATA_DIR / "movies.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"
LINKS_CSV = DATA_DIR / "links.csv"

@st.cache_data(show_spinner=False)
def download_and_extract_movielens():
    """Download ml-latest-small and extract movies.csv, ratings.csv, links.csv."""
    DATA_DIR.mkdir(exist_ok=True)
    if MOVIES_CSV.exists() and RATINGS_CSV.exists() and LINKS_CSV.exists():
        return
    r = requests.get(MOVIELENS_ZIP_URL, stream=True)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    for name in ["movies.csv", "ratings.csv", "links.csv"]:
        with z.open("ml-latest-small/" + name) as f_in:
            with open(DATA_DIR / name, "wb") as f_out:
                f_out.write(f_in.read())

@st.cache_data(show_spinner=False)
def load_movielens():
    download_and_extract_movielens()
    movies = pd.read_csv(MOVIES_CSV)
    ratings = pd.read_csv(RATINGS_CSV)
    links = pd.read_csv(LINKS_CSV)
    # Parse year from title if possible
    def parse_title_year(t):
        m = re.match(r"^(.*)\s+\((\d{4})\)$", t)
        if m:
            return m.group(1).strip(), int(m.group(2))
        else:
            return t, None
    movies[["clean_title", "year"]] = movies["title"].apply(lambda x: pd.Series(parse_title_year(x)))
    # Ensure genres list
    movies["genres_list"] = movies["genres"].fillna("").apply(lambda s: s.split("|") if s else [])
    # Add tmdbId if present in links
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    return movies, ratings

def ensure_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "ratings" not in st.session_state:
        st.session_state["ratings"] = {}  # movieId -> rating
    if "user_profile" not in st.session_state:
        st.session_state["user_profile"] = {"preferred_genres": [], "preferred_actors": [], "preferred_directors": []}

def parse_movie_query_title(user_input):
    """Try to parse a movie title from a user input — simple heuristics."""
    # If surrounded by quotes, return quoted text
    import re
    m = re.search(r'["“](.+?)["”]', user_input)
    if m:
        return m.group(1).strip()
    # If the user input matches "similar to X" patterns
    m = re.search(r'(?:similar to|like|similar)\s+(.+)', user_input, re.I)
    if m:
        return m.group(1).strip()
    # Else return None
    return None

OUT_OF_SCOPE_MESSAGE = "I’m sorry, but I can only discuss movies and cinema-related topics."

# smaller helper
def get_tmdb_key():
    return os.environ.get("TMDB_API_KEY", None)