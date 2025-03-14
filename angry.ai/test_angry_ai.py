import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model path
model_path = os.path.abspath("./angry_ai_model")
print(f"Loading model from: {model_path}")

# Load model and tokenizer
try:
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Angry phrases
angry_phrases = [
    "STOP WASTING MY TIME!",
    "UGH, HERE’S YOUR STUPID ANSWER!",
    "WHAT A DUMB QUESTION—PAY ATTENTION!",
    "FIX THIS YOURSELF NEXT TIME, IDIOT!",
]

def angry_response(text):
    return f"{random.choice(angry_phrases)} {text}"

def generate_solution(prompt):
    model.eval()
    # Tokenize with padding and attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    # Generate with sampling for variety
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Pass attention mask
        max_length=100,  # Allow longer output
        num_return_sequences=1,
        do_sample=True,  # Enable sampling
        temperature=0.7,  # Control randomness
        top_k=50,  # Limit to top 50 tokens for coherence
        pad_token_id=tokenizer.eos_token_id,  # Set pad token
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return angry_response(response.strip())

# Test it
while True:
    prompt = input("Ask me a Python question (or 'exit'): ")
    if prompt.lower() == "exit":
        break
    print(generate_solution(prompt))