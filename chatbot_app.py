import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- Configuration ---
MODEL_PATH = "./results_movie_chatbot/final_model"

# --- Model Loading ---
@st.cache_resource(show_spinner="Loading CinemaAI model...")
def load_model():
    """
    Loads the fine-tuned model and tokenizer from the specified path.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

# --- Page Configuration ---
st.set_page_config(layout="centered", page_title="CinemaAI Chatbot")
st.title("🎬 CinemaAI Chatbot")
st.write("This chatbot is fine-tuned on movie dialogues. Ask me anything!")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Response Generation ---
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the model input
    formatted_prompt = f"<s>{prompt}</s>"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    # Generate a response from the model
    with st.spinner("CinemaAI is thinking..."):
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            max_length=80,  # Controls the length of the response
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, # Enables more creative, less repetitive responses
            top_k=50,       # Considers the top 50 words for the next token
            top_p=0.95      # Uses nucleus sampling for better quality
        )
        
        # Decode and clean up the response
        response_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        # Remove the original prompt from the generated text
        clean_response = response_text.replace(prompt, "").strip()

    # Add assistant response to chat history
    with st.chat_message("assistant"):
        st.markdown(clean_response)
    st.session_state.messages.append({"role": "assistant", "content": clean_response})