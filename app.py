import os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr

# Retry mechanism for loading the model and tokenizer
RETRY_ATTEMPTS = 10
TIMEOUT = 20  # Increase the timeout to 20 seconds

# Use a pipeline as a high-level helper
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="microsoft/DialoGPT-large")
pipe(messages)

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

def predict(input_text):
    """Generate a response based solely on the current input text."""
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def chatbot(input_text):
    """Gradio chatbot function without multi-turn context."""
    response = predict(input_text)
    return [(input_text, response)]

# Create Gradio interface
iface = gr.Blocks()

with iface:
    gr.Markdown("# 🤖 Simple DialoGPT Chatbot")
    gr.Markdown("### A basic single-turn chatbot using DialoGPT")
    
    chatbot_interface = gr.Chatbot(label="Chat with AI")
    msg = gr.Textbox(label="Enter your message here")
    submit_btn = gr.Button("Submit")
    clear_btn = gr.Button("Clear Conversation")

    submit_btn.click(chatbot, [msg], chatbot_interface)
    clear_btn.click(lambda: [], None, chatbot_interface, queue=False)

iface.launch()
