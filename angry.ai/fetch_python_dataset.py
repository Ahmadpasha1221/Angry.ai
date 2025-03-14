from datasets import load_dataset
import os

# Create data folder
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Output file
output_file = os.path.join(data_dir, "python_dataset.txt")

# Load streaming dataset, filter for Python
ds = load_dataset("codeparrot/github-code", streaming=True, split="train",trust_remote_code=True)

# Collect Python code (limit to ~10MB or adjustable)
max_size_bytes = 1 * 1024 * 1024  # 10MB
current_size = 0
# trust_remote_code=True

with open(output_file, "w", encoding="utf-8") as f:
    for example in ds:
        if example["language"] == "Python":  # Filter for Python
            code = example["code"].strip()
            if code:  # Skip empty snippets
                f.write(code + "\n\n")  # Add separator between snippets
                current_size += len(code.encode("utf-8")) + 2  # Account for \n\n
                if current_size >= max_size_bytes:
                    break

print(f"Saved Python dataset to {output_file}. Size: ~{current_size / (1024 * 1024):.2f} MB")