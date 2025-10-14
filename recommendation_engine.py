# recommendation_engine.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import streamlit as st
from utils import load_movielens, ensure_session_state

@st.cache_data(show_spinner=False)
def build_content_matrix(movies):
    # Combine genres and title into a text field
    movies = movies.copy()
    movies["content"] = movies["genres"].fillna("") + " " + movies["clean_title"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english")
    content_matrix = tfidf.fit_transform(movies["content"])
    return tfidf, content_matrix

@st.cache_data(show_spinner=False)
def build_item_latent(ratings, n_components=50):
    # Build user-item matrix (sparse), perform truncated SVD to derive item latent features
    # Pivot ratings
    pivot = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    # If there are too many users or items, reduce
    svd = TruncatedSVD(n_components=min(n_components, min(pivot.shape)-1), random_state=42)
    latent = svd.fit_transform(pivot.T)  # items x components
    # Map movieId -> latent vector
    item_ids = pivot.columns.values
    item_latent_df = pd.DataFrame(latent, index=item_ids)
    return item_latent_df, svd

def build_user_profile_from_session(movies, ratings_session):
    # ratings_session: dict movieId -> rating
    if not ratings_session:
        return None
    # Weighted average of item content vectors
    _, content_matrix = build_content_matrix(movies)
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies["movieId"].values)}
    vectors = []
    weights = []
    for mid, r in ratings_session.items():
        if mid in movie_id_to_idx:
            vectors.append(content_matrix[movie_id_to_idx[mid]].toarray()[0])
            weights.append(r)
    if not vectors:
        return None
    weighted = np.average(np.vstack(vectors), axis=0, weights=np.array(weights))
    return weighted

def recommend_movies(movies, ratings_df, user_preferences=None, top_n=10):
    """
    Hybrid recommendations:
     - content similarity between user profile (from session ratings or preferences) and content vectors
     - collaborative component: similarity in latent item space aggregated
    """
    # Build content
    tfidf, content_matrix = build_content_matrix(movies)
    # Content-based score from preferences (simple TF-IDF query)
    if user_preferences and user_preferences.get("preferred_genres"):
        query = " ".join(user_preferences.get("preferred_genres", []))
        q_vec = tfidf.transform([query])
        content_scores = cosine_similarity(q_vec, content_matrix)[0]
    else:
        # If user has session ratings, build profile
        user_profile_vec = build_user_profile_from_session(movies, st.session_state.get("ratings", {}))
        if user_profile_vec is not None:
            content_scores = cosine_similarity([user_profile_vec], content_matrix)[0]
        else:
            content_scores = np.zeros(movies.shape[0])

    # Collaborative component
    try:
        item_latent_df, svd = build_item_latent(ratings_df)
        # Build user latent vector from session ratings if available
        if st.session_state.get("ratings"):
            # compute weighted average of item latent vectors
            vecs = []
            wts = []
            for mid, r in st.session_state["ratings"].items():
                if mid in item_latent_df.index:
                    vecs.append(item_latent_df.loc[mid].values)
                    wts.append(r)
            if vecs:
                user_latent = np.average(np.vstack(vecs), axis=0, weights=np.array(wts))
                # compute similarity to all items
                all_item_vecs = item_latent_df.values
                collab_scores_full = cosine_similarity([user_latent], all_item_vecs)[0]
                # Map collab scores to movies order
                collab_scores = np.zeros(movies.shape[0])
                movie_to_idx = {m: i for i, m in enumerate(movies["movieId"].values)}
                for idx, item_id in enumerate(item_latent_df.index):
                    if item_id in movie_to_idx:
                        collab_scores[movie_to_idx[item_id]] = collab_scores_full[idx]
            else:
                collab_scores = np.zeros(movies.shape[0])
        else:
            collab_scores = np.zeros(movies.shape[0])
    except Exception as e:
        # If collaborative fails (not enough data), fallback
        collab_scores = np.zeros(movies.shape[0])

    # Combine scores (weighted)
    combined = 0.6 * normalize_array(content_scores) + 0.4 * normalize_array(collab_scores)
    movies = movies.copy()
    movies["score"] = combined
    recs = movies.sort_values("score", ascending=False).head(top_n)
    return recs

def normalize_array(a):
    if a.max() == a.min():
        return np.zeros_like(a)
    return (a - a.min()) / (a.max() - a.min())

def find_similar_movies(movies, title, top_n=8):
    # Find movie by title fuzzy match
    from difflib import get_close_matches
    titles = movies["clean_title"].tolist()
    match = get_close_matches(title, titles, n=1)
    if not match:
        return pd.DataFrame()
    idx = movies[movies["clean_title"] == match[0]].index[0]
    tfidf, content_matrix = build_content_matrix(movies)
    sim = cosine_similarity(content_matrix[idx], content_matrix)[0]
    movies_copy = movies.copy()
    movies_copy["sim"] = sim
    res = movies_copy.sort_values("sim", ascending=False).iloc[1:top_n+1]  # exclude self
    return res
