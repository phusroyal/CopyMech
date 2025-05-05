from datasets import load_dataset

save_path = 'english_insertions'

# Load the dataset
dataset = load_dataset("google-research-datasets/wiki_atomic_edits", save_path)

print(f"Dataset saved to {save_path}")