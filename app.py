import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_DIR = "saved_model/gpt2_cinema/best" # This now points to your new, better model!

# MODEL LOADING
@st.cache_resource
def load_model():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
            
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

#RESPONSE GENERATION
def generate_response(tokenizer, model, user_prompt, chat_history):
    """
    Generate a clean response from the model.
    """
    full_prompt = ""
    for message in chat_history:
        if message["role"] == "user":
            full_prompt += f"User: {message['content']}\n"
        elif message["role"] == "assistant":
            full_prompt += f"Bot: {message['content']}\n"
    
    full_prompt += f"User: {user_prompt}\nBot:"

    inputs = tokenizer.encode(full_prompt, return_tensors='pt')
    
    # Generate output, stopping at the EOS token
    outputs = model.generate(
        inputs,
        max_length=len(inputs[0]) + 60, # Generate up to 60 new tokens
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, # <-- Tell the model to stop at the end token
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7,
        top_k=50
    )
    
    # Decode the response
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the bot's newly generated response
    bot_response_start = response_text.rfind("Bot:") + len("Bot:")
    bot_response = response_text[bot_response_start:].strip()
    
    return bot_response

#STREAMLIT UI
st.set_page_config(page_title="Film Recommender Chat", layout="centered")
st.title("🎬 Cinema Chatbot")
st.write("Ask me about movie directors, cast, release dates, or for recommendations!")

tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    if model is None:
        st.error("Model is not loaded. Cannot process request.")
    else:
        chat_history_for_prompt = st.session_state.messages.copy()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(tokenizer, model, prompt, chat_history_for_prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})