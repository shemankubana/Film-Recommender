# chatbot_app.py
import streamlit as st
from transformers import pipeline
import requests
import re
import os
from dotenv import load_dotenv

# --- Page Configuration ---
# This MUST be the first Streamlit command.
st.set_page_config(layout="centered", page_title="CinemaPedia")

load_dotenv()

# --- Configuration ---
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
QA_MODEL = "distilbert-base-cased-distilled-squad"

# --- Model & Helper Functions ---
@st.cache_resource(show_spinner="Loading CinemaPedia model...")
def load_qa_pipeline():
    """Loads the pre-trained Question Answering pipeline."""
    return pipeline("question-answering", model=QA_MODEL)

qa_pipeline = load_qa_pipeline()

def get_movie_context(movie_title):
    """Fetches movie plot summary from TMDB to use as context."""
    if not TMDB_API_KEY:
        return None, "TMDB_API_KEY is not set. Please set it to fetch movie plots."

    search_url = f"https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_title}
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        
        if not results:
            return None, f"Sorry, I couldn't find a movie called '{movie_title}'."
            
        movie_title = results[0]["title"]
        overview = results[0]["overview"]
        
        if not overview:
            return None, f"I found '{movie_title}', but it doesn't have a plot summary for me to read."

        return overview, None
    except requests.exceptions.RequestException as e:
        return None, f"An error occurred while fetching movie data: {e}"

def extract_movie_title(query):
    """Extracts a movie title from the user's query using regex."""
    match = re.search(r'["“](.+?)["”]', query)
    if match:
        return match.group(1)
    match = re.search(r'(?:in|about|of) the movie\s+(.+)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip("?")
    return None

# --- UI & Interaction Logic ---
st.title("🎬 CinemaPedia")
st.write("Your factual assistant for movie knowledge. Ask me a question about a film!")

if not TMDB_API_KEY:
    st.warning("Please provide a TMDB_API_KEY in a .env file to enable this app's full functionality.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me something like: 'What is the plot of \"Inception\"?'"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching for answers..."):
            movie_title = extract_movie_title(prompt)
            
            if not movie_title:
                bot_response = "Please mention a movie title in quotes (e.g., \"The Matrix\") for me to look up."
            else:
                context, error = get_movie_context(movie_title)
                if error:
                    bot_response = error
                else:
                    result = qa_pipeline(question=prompt, context=context)
                    bot_response = result['answer'].capitalize()
        
        st.markdown(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})