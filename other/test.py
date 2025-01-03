import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the GPT-2 XL tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
model.to(device)
model.eval()

# Create a long text with a repeated span
text = (
    "Once upon a time in a land far, far away, there lived a wise old king. "
    "He ruled his kingdom with justice and kindness, and all his subjects loved him dearly. "
    "One day, a young traveler arrived at the kingdom. He had heard tales of the king's wisdom and sought his counsel. "
    "The wise old king welcomed the traveler warmly. "
    "He ruled his kingdom with justice and kindness, and all his subjects loved him dearly. "  # Repeated span
    "The traveler was amazed by the prosperity and happiness of the people. "
    "He decided to stay and learn from the king. "
    "Years passed, and the traveler became a trusted advisor to the king. "
)

# Tokenize the text
input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

# Convert input_ids to list for easy manipulation
tokens = input_ids[0].tolist()

# Decode tokens for verification
decoded_tokens = [tokenizer.decode([token]) for token in tokens]

# Manually identify the positions of the repeated spans
# First occurrence
t1_start = 19
t1_end = 34  # Adjust if necessary

# Second occurrence
t2_start = 71
t2_end = 86  # Adjust if necessary

# Verify that the tokens in both spans are the same
print("\nFirst occurrence tokens:")
for i in range(t1_start, t1_end + 1):
    print(f"Token {i}: {tokens[i]} -> {repr(decoded_tokens[i])}")

print("\nSecond occurrence tokens:")
for i in range(t2_start, t2_end + 1):
    print(f"Token {i}: {tokens[i]} -> {repr(decoded_tokens[i])}")

# Function to adjust hidden states
def adjust_hidden_states(module, input, output):
    # output is a tuple: (hidden_states, presents)
    hidden_states = output[0]  # Extract the hidden states tensor
    # Copy the hidden states from the first occurrence to the second occurrence
    hidden_states[:, t2_start:t2_end+1, :] = hidden_states[:, t1_start:t1_end+1, :]
    # Return the modified output as a tuple
    return (hidden_states,) + output[1:]

# Register the hook on the transformer blocks
hooks = []
for idx, block in enumerate(model.transformer.h):
    hook = block.register_forward_hook(adjust_hidden_states)
    hooks.append(hook)

# Run the model with the hooks (adjusted)
with torch.no_grad():
    outputs_adjusted = model(input_ids)
    logits_adjusted = outputs_adjusted.logits

# Remove hooks after use
for hook in hooks:
    hook.remove()

# Run the model without adjustments for comparison
with torch.no_grad():
    outputs_original = model(input_ids)
    logits_original = outputs_original.logits

# Compare the logits of the token following the repeated span
for i in range(t2_end + 1, t2_end + 10):
    next_token_position = t2_end + i
    logits_adjusted_next = logits_adjusted[:, next_token_position, :]
    logits_original_next = logits_original[:, next_token_position, :]

    # Get the top predicted tokens
    top_k = 5
    prob_original = torch.softmax(logits_original_next, dim=-1)
    prob_adjusted = torch.softmax(logits_adjusted_next, dim=-1)

    top_tokens_original = torch.topk(prob_original, top_k, dim=-1)
    top_tokens_adjusted = torch.topk(prob_adjusted, top_k, dim=-1)

    print("\nTop predictions after the adjusted repeated span:")
    for idx in top_tokens_adjusted.indices[0]:
        print(tokenizer.decode([idx.item()]))

    print("\nTop predictions after the original repeated span:")
    for idx in top_tokens_original.indices[0]:
        print(tokenizer.decode([idx.item()]))

    # Check if the predictions are the same
    are_predictions_same = torch.equal(top_tokens_original.indices, top_tokens_adjusted.indices)
    print(f"\nAre the top {top_k} predictions the same after adjustment? {are_predictions_same}")
