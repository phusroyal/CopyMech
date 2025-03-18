from datasets import load_from_disk
from tqdm import tqdm
import random, re

def wiki_loader(num_samples = 1e6):
    # Load the dataset from disk
    subset = load_from_disk("/home/longnhat/workspace_phu/CopyMech/english_insertions")

    base_sents = subset['train']['base_sentence'][:num_samples]
    phrases = subset['train']['phrase'][:num_samples]
    edited_sents = subset['train']['edited_sentence'][:num_samples]

    return base_sents, phrases, edited_sents

def template_searcher(seq: str, num_context = 26, num_post_word = 22,  source = 'is', target = 'are') -> bool:
    """
    Returns True if the sequence has an occurrence of 'is' or 'are'
    that is preceded (anywhere earlier in the sequence) by at least 6 tokens that are exactly 'space'.
    
    Examples:
      'There space space space space space oh space is a cat.' -> True
      'There space are many cats.' -> False
      'There is a cat.' -> False
      'There space space space space space space is a cat.' -> True
      'There spaces are many cats.' -> False
    """
    tokens = seq.split()
    # check if sentence only has 1 is or are
    if tokens.count(source) + tokens.count(target) != 1:
        return False
    for i, token in enumerate(tokens):
        if token in {source, target}:
            # Count how many tokens before this occurrence are exactly "space"
            if len(tokens[:i]) >= num_context and len(tokens[i:]) >= num_post_word:
                return True
    return False

def text_swap(text, text1 = 'is', text2 = 'are'):
    """Given a text, replace ' is ' by ' are ', and vice versa. 
    Return the corrupted text, and the text until the first is/are."""
    _text1 = f' {text1} '
    _text2 = f' {text2} '

    text = text.strip()
    if _text1 in text:
        corrupted_text = text.replace(_text1, _text2, 1)
    elif _text2 in text:
        corrupted_text = text.replace(_text2, _text1, 1)
    
    # find position of first is/are and return text before that
    first_text1 = text.find(_text1)
    first_text2 = text.find(_text2)
    if first_text1 == -1 and first_text2 == -1:
        return None
    elif first_text1 == -1:
        return corrupted_text, text[:first_text2], text2
    elif first_text2 == -1:
        return corrupted_text, text[:first_text1], text1
    
    return corrupted_text, text[:min(first_text1, first_text2)]

def text_drop(text, drop = 'a', replace = ''):
    """Given a text, drop ' a ' by ''. 
    Return the corrupted text, and the text until the first 'a'."""
    _drop = f' {drop} '
    _replace = f' {replace} '

    text = text.strip()
    if _drop in text:
        corrupted_text = text.replace(_drop, _replace, 1)
        # rm multiple spaces
        corrupted_text = re.sub(r"\s+", " ", corrupted_text).strip()
    
    # find position of first is/are and return text before that
    first_drop = text.find(_drop)
    if first_drop == -1:
        return None
    elif first_drop != -1:
        return corrupted_text, text[:first_drop], drop

def modify_token(token: str, method: str = None) -> str:
    """
    Modify a token using one of three methods:
      - "swap": swap the first two characters (if possible)
      - "drop": drop the last character
      - "add": insert a random lowercase letter before the last character
    If method is None, choose one at random.
    """
    if method is None:
        method = random.choice(["swap", "drop", "add"])
    
    if method == "swap":
        if len(token) < 2:
            return token
        # Swap the first two characters.
        return token[1] + token[0] + token[2:]
    elif method == "drop":
        if len(token) < 1:
            return token
        # Drop the last character.
        return token[:-1]
    elif method == "add":
        if len(token) < 1:
            return token
        # Insert a random lowercase letter before the last character.
        letter = random.choice("abcdefghijklmnopqrstuvwxyz")
        return token[:-1] + letter + token[-1]
    else:
        raise ValueError(f"Unknown method: {method}")

def corrupt_and_compare_tokens(orig_tokens, target_token, method: str = None):
    """
    Given a list of tokens (e.g. ['i','have','a','ham','burger']) and a target token to corrupt,
    modify the target token using one of three methods ("swap", "drop", or "add").
    
    When corrupting, if the token is not the first token, remove the intervening space between it and
    its preceding token. (E.g., merging "ham" and the modified "burger" into "hamubrger".)
    
    Then, simulate tokenization of the corrupted sentence as follows:
      - If the corruption merged two tokens, we assume that a subword tokenizer would split it
        into [previous_token, modified_token]. Otherwise, the corrupted token is as produced.
    
    Finally, compare the original token list to the corrupted token list (simulated)
    to find the first token where they differ. Return:
      1. The decoded corrupted sentence up to (but not including) the mismatched token.
      2. The full corrupted sentence (string).
      3. The next ground-truth token (from the original) at the mismatch position.
    
    Example:
      orig_tokens = ['i', 'have', 'a', 'ham', 'burger']
      target_token = 'burger'
      If method "swap" is used, modified token becomes "ubrger". Then the corrupted sentence is:
         "i have a hamubrger"
      which we simulate as tokenized to:
         ['i', 'have', 'a', 'ham', 'ubrger']
      The first mismatch is at index 4 (comparing to ['i','have','a','ham','burger']),
      so we return:
         ("i have a ham", "i have a hamubrger", "burger")
    """
    # Find the target token index (assume first occurrence)
    try:
        target_idx = orig_tokens.index(target_token)
    except ValueError:
        raise ValueError("Target token not found in the original tokens.")
    
    # Modify the token using the specified method.
    modified_token = modify_token(target_token, method=method)
    
    # Build the corrupted tokens.
    # If the target token is not the first token, merge it with the preceding token (i.e. drop the space).
    corrupted_tokens = orig_tokens.copy()
    if target_idx == 0:
        corrupted_tokens[0] = modified_token
    else:
        # Merge the preceding token with the modified token.
        # For display, we remove the space between them.
        merged = corrupted_tokens[target_idx - 1] + modified_token
        # Replace the preceding token with the merged token and remove the target token.
        corrupted_tokens[target_idx - 1] = merged
        corrupted_tokens.pop(target_idx)
    
    # Build the corrupted sentence (joined by spaces).
    corrupted_sentence = " ".join(corrupted_tokens)
    
    # Simulate tokenization of the corrupted sentence.
    # If the target was not at index 0, assume the merged token gets split back into two tokens:
    # the original preceding token and the modified token.
    if target_idx > 0:
        simulated_tokens = (
            orig_tokens[:target_idx - 1] +
            [orig_tokens[target_idx - 1], modified_token] +
            orig_tokens[target_idx+1:]
        )
    else:
        simulated_tokens = corrupted_tokens  # no merging happened
    
    # Compare the original tokens with the simulated corrupted tokens to find the first mismatch.
    min_len = min(len(orig_tokens), len(simulated_tokens))
    mismatch_index = None
    for i in range(min_len):
        if orig_tokens[i] != simulated_tokens[i]:
            mismatch_index = i
            break
    if mismatch_index is None:
        mismatch_index = min_len  # if all tokens match in the common prefix
    
    # The decoded corrupted sentence up to the mismatch (join tokens from simulated list up to mismatch).
    decoded_up_to = " ".join(simulated_tokens[:mismatch_index])
    
    # The next ground-truth token from the original (if available).
    ground_truth_next = orig_tokens[mismatch_index] if mismatch_index < len(orig_tokens) else None
    
    return decoded_up_to, corrupted_sentence, ground_truth_next

def swap_is_are(text):
    source, target = 'is', 'are'

    if not template_searcher(seq= text,
                      num_context= 26,
                      num_post_word= 22,
                      source= source,
                      target= target):
        return None
    
    return text_swap(text= text,
                     text1= source,
                     text2= target)

def swap_was_were(text):
    source, target = 'was', 'were'

    if not template_searcher(seq= text,
                      num_context= 26,
                      num_post_word= 22,
                      source= source,
                      target= target):
        return None
    
    return text_swap(text= text,
                     text1= source,
                     text2= target)

def swap_a_the(text):
    source, target = 'a', 'the'

    if not template_searcher(seq= text,
                      num_context= 26,
                      num_post_word= 22,
                      source= source,
                      target= target):
        return None
    
    return text_swap(text= text,
                     text1= source,
                     text2= target)

def drop_text(text, drop = 'a', replace = ''):
    if not template_searcher(seq= text,
                      num_context= 26,
                      num_post_word= 22,
                      source= drop,
                      target= replace):
        return None
    
    return text_drop(text= text,
                     drop= drop,
                     replace= replace)

def char_edit(text, model, needed_len = 47, target_id = 26):
    # tokenize the token
    tokens = model.to_tokens(text, prepend_bos = False)
    if len(tokens[0]) < needed_len:
        return None

    # select the word id
    target_token = tokens[:, target_id]

    return_outputs = []
    for method in ["swap", "drop", "add"]:
        output = corrupt_and_compare_tokens(tokens, 
                                            target_token, 
                                            method=method)
        return_outputs.append(output)

    return return_outputs


# # Example usage:
# orig_tokens = ['i', 'have', 'a', 'chic', 'ken']
# target_token = 'ken'

# # Try each method:
# for method in ["swap", "drop", "add"]:
#     decoded, corrupted, gt_next = corrupt_and_compare_tokens(orig_tokens, target_token, method=method)
#     print(f"Method: {method}")
#     print("Decoded up to mismatch:", decoded)
#     print("Corrupted sentence:", corrupted)
#     print("Next ground truth token:", gt_next)
#     print("-" * 50)
