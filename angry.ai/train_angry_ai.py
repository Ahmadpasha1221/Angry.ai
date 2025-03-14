import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Tokenizer and pre-trained model
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

# Adjust model size for 2GB GPU
model.config.n_layer = 4  # Reduce to 4 layers
# No need to manually adjust head_mask—it’s optional and defaults to None

# Load dataset
dataset = load_dataset("text", data_files={"train": "data/python_dataset.txt"})

def tokenize_function(examples):
    encodings = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Training args
training_args = TrainingArguments(
    output_dir="./angry_ai_model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Train
trainer.train()

# Save
model.save_pretrained("./angry_ai_model")
tokenizer.save_pretrained("./angry_ai_model")