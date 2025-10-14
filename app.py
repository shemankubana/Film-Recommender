# app.py
import streamlit as st
from utils import ensure_session_state, load_movielens, OUT_OF_SCOPE_MESSAGE, parse_movie_query_title, get_tmdb_key
from recommendation_engine import recommend_movies, find_similar_movies
from trivia_module import answer_trivia
import pandas as pd
import requests
import os
import re

# --- Helper: TMDB poster fetch ---
def get_poster_path_from_tmdb(tmdb_id):
    key = os.environ.get("TMDB_API_KEY")
    if not key:
        return None
    r = requests.get(f"https://api.themoviedb.org/3/movie/{tmdb_id}", params={"api_key": key})
    if r.status_code != 200:
        return None
    data = r.json()
    return data.get("poster_path")

# --- Page config ---
st.set_page_config(layout="wide", page_title="CinemaAI — Movie Assistant")

# --- Session State Initialization ---
ensure_session_state()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "submitted" not in st.session_state:
    st.session_state.submitted = False

# --- Load data ---
movies, ratings_df = load_movielens()
# Convert list columns to tuples to avoid caching/hash issues
movies["genres_list"] = movies["genres_list"].apply(lambda x: tuple(x) if isinstance(x, list) else x)

# --- Sidebar ---
with st.sidebar:
    st.header("CinemaAI — Preferences")
    genres = sorted({g for sub in movies["genres_list"] for g in sub if g})
    chosen_genres = st.multiselect("Preferred genres (optional)", genres, key="pref_genres")
    st.write("Session ratings stored locally (cleared on page reload).")
    if st.button("Clear session ratings"):
        st.session_state["ratings"] = {}
        st.success("Session ratings cleared.")

# --- Main UI ---
st.title("🎬 CinemaAI — Intelligent Movie Assistant")
st.write("Ask for movie recommendations, similar movies, trivia, or rate movies. Out-of-scope questions will be politely refused.")

# --- Helper to add messages ---
def add_message(sender, text):
    st.session_state.messages.append({"sender": sender, "text": text})

# --- Display chat messages ---
chat_col1, chat_col2 = st.columns([3, 1])
with chat_col1:
    st.subheader("Chat")
    for msg in st.session_state.messages:
        if msg["sender"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**CinemaAI:** {msg['text']}")

# --- Input ---
with chat_col1:
    user_input = st.text_input("Your message", key="user_input")
    submitted = st.button("Send")

# --- Quick actions ---
with chat_col2:
    st.subheader("Quick Actions")
    if st.button("Recommend movies for me"):
        user_input = "recommend me movies"
        submitted = True
    if st.button("Top picks (popular)"):
        user_input = "popular movies"
        submitted = True
    api_key_present = bool(get_tmdb_key())
    if not api_key_present:
        st.info("Add TMDB API key in env var TMDB_API_KEY for verified trivia and posters (optional).")

# --- Processing input ---
if submitted and user_input:
    st.session_state.submitted = True

if st.session_state.submitted:
    add_message("user", user_input)
    lower = user_input.lower()

    # --- Scope detection ---
    movie_vocab = ["movie", "film", "watch", "director", "actor", "actors", "cast",
                   "genre", "rating", "ratings", "similar", "recommend", "recommendation",
                   "trivia", "plot", "release", "when", "who", "what", "where", "which"]
    if not any(tok in lower for tok in movie_vocab):
        bot_reply = OUT_OF_SCOPE_MESSAGE
        add_message("bot", bot_reply)

    # --- Handle ratings ---
    m = re.search(r"rate\s+['\"]?(.+?)['\"]?\s+([1-5])", user_input, re.I)
    if m:
        title = m.group(1).strip()
        rating_val = int(m.group(2))
        cand = movies[movies["clean_title"].str.lower().str.contains(title.lower())]
        if not cand.empty:
            mid = int(cand.iloc[0]["movieId"])
            st.session_state["ratings"][mid] = rating_val
            bot_reply = f"Thanks — recorded a rating of {rating_val} for {cand.iloc[0]['clean_title']}."
        else:
            bot_reply = f"Couldn't find a movie titled '{title}' in local dataset."
        add_message("bot", bot_reply)

    # --- Recommendations ---
    if any(w in lower for w in ["recommend", "suggest", "top picks", "popular"]):
        user_prefs = {"preferred_genres": st.session_state.get("pref_genres", [])}
        recs = recommend_movies(movies, ratings_df, user_preferences=user_prefs, top_n=8)
        bot_text = "Here are some recommendations based on your session preferences and ratings:"
        add_message("bot", bot_text)
        cols = st.columns(4)
        for i, (_, row) in enumerate(recs.iterrows()):
            c = cols[i % 4]
            with c:
                title = f"{row['clean_title']} ({int(row['year'])})" if not pd.isna(row['year']) else row['clean_title']
                st.markdown(f"**{title}**")
                # Poster fetching
                poster_shown = False
                tmdb_id = row.get("tmdbId", None)
                if tmdb_id and os.environ.get("TMDB_API_KEY"):
                    poster = get_poster_path_from_tmdb(int(tmdb_id))
                    if poster:
                        st.image(f"https://image.tmdb.org/t/p/w300{poster}", use_container_width=True)
                        poster_shown = True
                if not poster_shown:
                    st.write("_No poster available_")
                st.write(", ".join(row["genres_list"][:3]))
                if row.get("score") is not None:
                    st.write(f"Score: {row['score']:.2f}")
                # rating widget
                rkey = f"rate_{row['movieId']}"
                val = st.radio("Rate", options=[1,2,3,4,5], index=2, key=rkey, horizontal=True)
                if st.button("Save rating", key=f"save_{row['movieId']}"):
                    st.session_state["ratings"][int(row['movieId'])] = val
                    st.success(f"Saved rating {val} for {row['clean_title']}")

    # --- Similar movies ---
    if any(w in lower for w in ["similar to", "similar", "like"]):
        parsed = parse_movie_query_title(user_input)
        if not parsed:
            add_message("bot", "Please specify the movie title (e.g., Similar to 'Inception').")
        else:
            sims = find_similar_movies(movies, parsed, top_n=6)
            if sims.empty:
                add_message("bot", f"Couldn't find similar movies to '{parsed}' in local dataset.")
            else:
                add_message("bot", f"Here are movies similar to '{parsed}':")
                cols = st.columns(3)
                for i, (_, row) in enumerate(sims.iterrows()):
                    c = cols[i % 3]
                    with c:
                        st.markdown(f"**{row['clean_title']} ({int(row['year']) if not pd.isna(row['year']) else ''})**")
                        st.write(", ".join(row["genres_list"][:3]))
                        st.write(f"Similarity: {row['sim']:.2f}")
                        if st.button("Rate", key=f"sim_rate_{row['movieId']}"):
                            st.session_state["ratings"][int(row['movieId'])] = 4
                            st.success(f"Saved rating 4 for {row['clean_title']}")

    # --- Trivia questions ---
    if any(w in lower for w in ["who", "what", "when", "which", "where", "how long", "runtime", "released"]):
        trivia_answer = answer_trivia(user_input)
        add_message("bot", trivia_answer)

    # --- Default fallback ---
    if not any(w in lower for w in ["recommend", "similar", "rate", "who", "what", "when", "how long", "runtime", "released"]):
        add_message("bot", "I can help with recommendations, similar movies, ratings, and verified trivia. Try asking: 'Recommend me sci-fi movies' or 'Who directed \"Inception\"?'")

    # Reset submission state
    st.session_state.submitted = False