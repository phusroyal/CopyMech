from datasets import load_dataset

import os
os.environ["https_proxy"] = "http://xen03.iitd.ac.in:3128"
os.environ["http_proxy"] = "http://xen03.iitd.ac.in:3128"

token = '_'

os.environ["HUGGINGFACE_HUB_TOKEN"] = token
os.environ["HF_TOKEN"] = token  # Alternative variable name

# Load the dataset
dataset = load_dataset("google-research-datasets/wiki_atomic_edits", 'english_insertions', 
          cache_dir='./datasets/')
