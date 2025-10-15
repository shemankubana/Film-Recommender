"""
Simple Gradio UI to interact with the fine-tuned GPT-2 model.
Run: python webapp/gradio_app.py --model_dir saved_model/gpt2_cinema/best
"""

import os
import argparse
import evaluate.webapp.gradio_app as gr
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf

def load_model(model_dir):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = TFGPT2LMHeadModel.from_pretrained(model_dir)
    return tokenizer, model

def respond(tokenizer, model, user_input, max_length=100):
    prompt = f"User: {user_input}  Bot:"
    input_ids = tokenizer.encode(prompt, return_tensors="tf")
    gen = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    if prompt in out:
        out = out.split(prompt, 1)[1].strip()
    return out

def main(model_dir="saved_model/gpt2_cinema/best", port=7860):
    tokenizer, model = load_model(model_dir)
    def chat_fn(user_text):
        return respond(tokenizer, model, user_text)
    with gr.Blocks() as demo:
        gr.Markdown("# Cinema Chatbot")
        txt = gr.Textbox(lines=2, placeholder="Ask me about movies, actors, or ask for recommendations...")
        out = gr.Textbox(lines=6)
        btn = gr.Button("Send")
        btn.click(chat_fn, inputs=txt, outputs=out)
    demo.launch(server_name="0.0.0.0", server_port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="saved_model/gpt2_cinema/best")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(model_dir=args.model_dir, port=args.port)
