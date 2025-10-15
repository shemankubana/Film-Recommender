import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM
import requests
import re
import os
from dotenv import load_dotenv

# Page Configuration - MUST be first Streamlit command
st.set_page_config(layout="centered", page_title="CinemaPedia - Movie Chatbot")

load_dotenv()

# Configuration
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
MODEL_PATH = "./results_movie_chatbot/final_model"

# Check if fine-tuned model exists
if not os.path.exists(MODEL_PATH):
    st.error("Fine-tuned model not found! Please run 'fine_tune_model.py' first.")
    st.stop()

# Load Model and Tokenizer
@st.cache_resource(show_spinner="Loading TensorFlow model...")
def load_model():
    """Load the fine-tuned TensorFlow model and tokenizer"""
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Load TensorFlow model
        model = TFAutoModelForCausalLM.from_pretrained(MODEL_PATH)
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        st.success("✓ TensorFlow model loaded successfully!")
        return tokenizer, model
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure you've run 'fine_tune_model.py' with TensorFlow first.")
        st.stop()

tokenizer, model = load_model()

# Helper Functions
def get_movie_context(movie_title):
    """Fetch movie plot summary from TMDB"""
    if not TMDB_API_KEY:
        return None, "TMDB_API_KEY not set. Please add it to .env file."
    
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_title}
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        
        if not results:
            return None, f"Sorry, I couldn't find '{movie_title}'."
        
        movie_title = results[0]["title"]
        overview = results[0]["overview"]
        
        if not overview:
            return None, f"Found '{movie_title}', but no plot summary available."
        
        return overview, None
    
    except Exception as e:
        return None, f"Error fetching movie data: {e}"

def extract_movie_title(query):
    """Extract movie title from user query"""
    # Check for quotes
    match = re.search(r'[""](.+?)[""]', query)
    if match:
        return match.group(1)
    
    # Check for patterns like "about X" or "of X"
    match = re.search(r'(?:about|of) the movie\s+(.+)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip("?")
    
    return None

def is_movie_related(query):
    """Check if query is movie-related"""
    movie_keywords = [
        'movie', 'film', 'cinema', 'actor', 'actress', 'director',
        'plot', 'genre', 'recommend', 'watch', 'scene', 'character',
        'cast', 'starring', 'released', 'sequel', 'franchise'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in movie_keywords)

def generate_response(prompt, max_length=100):
    """Generate response using TensorFlow model"""
    try:
        # Format input
        input_text = f"<s>{prompt}</s>"
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            return_tensors="tf",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Generate with TensorFlow
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response (remove prompt if it's repeated)
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response if response else "I'm not sure how to respond to that."
    
    except Exception as e:
        return f"Error generating response: {e}"

# UI
st.title("🎬 CinemaPedia - Movie Chatbot")
st.markdown("Your AI assistant for all things movies! Powered by TensorFlow.")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot is fine-tuned using **TensorFlow** to discuss:
    - Movie recommendations
    - Plot summaries
    - Actors and directors
    - Movie trivia
    
    **Model:** DistilGPT-2 (fine-tuned)
    **Framework:** TensorFlow
    """)
    
    if TMDB_API_KEY:
        st.success("✓ TMDB API connected")
    else:
        st.warning("⚠️ TMDB API not configured")
    
    st.markdown("---")
    st.markdown("**Example questions:**")
    st.code("What's a good sci-fi movie?")
    st.code("Tell me about \"Inception\"")
    st.code("Who directed The Matrix?")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your movie chatbot powered by TensorFlow. Ask me anything about movies!"
        }
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about movies..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Check if movie-related
            if not is_movie_related(prompt):
                bot_response = "I'm a movie chatbot! I can only discuss films, actors, directors, and cinema. Please ask me about movies."
            else:
                # Try to extract movie title for TMDB lookup
                movie_title = extract_movie_title(prompt)
                
                if movie_title and TMDB_API_KEY:
                    context, error = get_movie_context(movie_title)
                    if context:
                        # Use context + model for better response
                        enhanced_prompt = f"About the movie {movie_title}: {context[:200]}... User asks: {prompt}"
                        bot_response = generate_response(enhanced_prompt, max_length=150)
                    else:
                        bot_response = generate_response(prompt)
                else:
                    # Use fine-tuned model directly
                    bot_response = generate_response(prompt)
            
            st.markdown(bot_response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

# Footer
st.markdown("---")
st.caption("Built with TensorFlow & Hugging Face Transformers")