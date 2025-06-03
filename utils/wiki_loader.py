from datasets import load_dataset

# Load the dataset
dataset = load_dataset("google-research-datasets/wiki_atomic_edits", 'english_insertions', 
          cache_dir='./datasets/')
