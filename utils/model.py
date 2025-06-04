# Import stuff

import torch
torch.set_grad_enabled(False)

from transformer_lens import HookedTransformer

# load model and tokenizer
model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-3B")
