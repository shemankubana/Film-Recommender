# trivia_module.py
import requests
from utils import get_tmdb_key, OUT_OF_SCOPE_MESSAGE
import streamlit as st
from utils import load_movielens
import pandas as pd
import re

TMDB_BASE = "https://api.themoviedb.org/3"

def _query_tmdb_by_title(title):
    key = get_tmdb_key()
    if not key:
        return None
    r = requests.get(f"{TMDB_BASE}/search/movie", params={"api_key": key, "query": title})
    r.raise_for_status()
    data = r.json()
    if data["results"]:
        return data["results"][0]  # best match
    return None

def get_tmdb_movie_details(tmdb_id):
    key = get_tmdb_key()
    if not key:
        return None
    r = requests.get(f"{TMDB_BASE}/movie/{tmdb_id}", params={"api_key": key})
    r.raise_for_status()
    details = r.json()
    # get credits for director/actors
    cr = requests.get(f"{TMDB_BASE}/movie/{tmdb_id}/credits", params={"api_key": key})
    cr.raise_for_status()
    credits = cr.json()
    return details, credits

def answer_trivia(question):
    """
    Only answer trivia if it can be verified:
      - If TMDB key present: fetch TMDB and answer (director, release_date, runtime, cast)
      - Else: only answer what local MovieLens dataset contains (title and year parsing).
    Avoid hallucinations: if unsure, reply with safe refusal.
    """
    movies, ratings = load_movielens()
    key = get_tmdb_key()

    # quick normalization
    q = question.lower()

    # Try to extract a movie title from question heuristics
    m = re.search(r'["“](.+?)["”]', question)
    title = None
    if m:
        title = m.group(1)
    else:
        # look for 'who directed X' pattern
        m2 = re.search(r'(?:who directed|director of)\s+(.+)', q)
        if m2:
            title = m2.group(1).strip().rstrip("?")
        # 'what year did TITANIC release' pattern
        m3 = re.search(r'what year did\s+(.+?)\s+release', q)
        if m3:
            title = m3.group(1).strip()
    # If we couldn't detect a title, decline
    if not title:
        return "I could not identify a movie title in your question. Please include the movie title (e.g., \"Inception\")."

    # If TMDB key present, use TMDB to answer robustly
    if key:
        try:
            search = _query_tmdb_by_title(title)
            if not search:
                return f"I couldn't find a verified record for '{title}' in TMDB."
            tmdb_id = search["id"]
            details, credits = get_tmdb_movie_details(tmdb_id)
            # Now answer depending on the question intent
            if re.search(r'director|who directed|who is the director', q):
                dirs = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
                if dirs:
                    return f"The director of {details.get('title')} ({details.get('release_date', '')[:4]}) is {', '.join(dirs)}."
                else:
                    return f"TMDB doesn't list a director for {details.get('title')}."
            if re.search(r'what year|release year|when did .* release', q):
                rd = details.get("release_date")
                if rd:
                    return f"{details.get('title')} was released on {rd}."
                else:
                    return f"TMDB does not have a release date listed for {details.get('title')}."
            if re.search(r'runtime|how long', q):
                rt = details.get("runtime")
                if rt:
                    return f"{details.get('title')} runtime is {rt} minutes."
                else:
                    return f"TMDB does not list a runtime for {details.get('title')}."
            if re.search(r'cast|starring|who starred|actors', q):
                cast = credits.get("cast", [])[:6]
                names = [c["name"] for c in cast]
                if names:
                    return f"The top billed cast of {details.get('title')} includes: {', '.join(names)}."
                else:
                    return f"TMDB doesn't list cast information for {details.get('title')}."
            # default fallback: provide a short verified summary from TMDB
            overview = details.get("overview")
            if overview:
                return f"Verified (TMDB) summary for {details.get('title')}: {overview}"
            else:
                return f"I found {details.get('title')} on TMDB but couldn't extract the requested fact."
        except Exception as e:
            return f"An error occurred while querying TMDB: {str(e)}"

    # No TMDB key: fall back to local dataset (only safe facts we can extract)
    # Find movie in local movies by title (clean_title match)
    candidates = movies[movies["clean_title"].str.lower().str.contains(title.lower())]
    if candidates.empty:
        # try exact title
        candidates = movies[movies["clean_title"].str.lower() == title.lower()]
    if candidates.empty:
        return f"I cannot verify facts about '{title}' with local data. Provide a TMDB API key to enable verified trivia."

    movie = candidates.iloc[0]
    # If user asked for year
    if re.search(r'what year|release year|when did .* release', q):
        if movie["year"]:
            return f"According to the local MovieLens data, {movie['clean_title']} was released in {int(movie['year'])}."
        else:
            return f"Release year not available for {movie['clean_title']} in the local dataset."

    # For director/actors we must avoid hallucination
    if re.search(r'director|who directed|who is the director|actor|cast|starring', q):
        return "I don’t have verified information for that question in the local dataset. Provide a TMDB API key to enable verified trivia."

    # default
    return "I cannot answer that question with verified local data. Provide a TMDB API key to enable more trivia."
