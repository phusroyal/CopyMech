from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import random, re

def wiki_loader(num_samples = 1000000):
    # Load the dataset from disk
    subset = load_dataset("google-research-datasets/wiki_atomic_edits", 'english_insertions', 
          cache_dir='./datasets/')

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
    
    return None

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

def modify_token(token: str, method: str = None, seed = 555) -> str:
    """
    Modify a token using one of three methods:
      - "swap": swap the first two characters (if possible)
      - "drop": drop the last character
      - "add": insert a random lowercase letter before the last character
    If method is None, choose one at random.
    """
    if len(token) < 3:
        return None
    
    if method == "swap":
        random.seed(seed)
        tokens = list(token)
        if len(tokens) < 3:
            return None
        i, j = random.sample(range(1, len(tokens)), 2) # 1 for cases with space in the beginning
        tokens[i], tokens[j] = tokens[j], tokens[i]
        return ''.join(tokens)
    elif method == "drop":
        random.seed(seed)
        tokens = list(token)
        j = random.sample(range(2,len(tokens)), 1)[0]
        tokens = tokens[:j-1] + tokens[j:]
        return ''.join(tokens)
    elif method == "add":
        seed = random.randint(1, 99)
        random.seed(seed)

        tokens = list(token)
        j = random.sample(range(1,len(tokens)), 1)[0]
        letter = random.choice("abcdefghijklmnopqrstuvwxyz")
        tokens = tokens[:j] + [letter] + tokens[j:]
        return ''.join(tokens)
    else:
        raise ValueError(f"Unknown method: {method}")

def get_outputs_modify(orig_tokens, target_token, idx_target, method: str = None):
    # Modify the token using the specified method.
    modified_token = modify_token(target_token, method=method)

    if modified_token is None:
        return None
    
    # Build the corrupted tokens.
    corrupted_tokens = orig_tokens[:idx_target] + [modified_token] + orig_tokens[idx_target+1:]

    # Build the corrupted sentence
    corrupted_sentence = "".join(corrupted_tokens)

    # The decoded corrupted sentence up to the target_token
    decoded_up_to = "".join(orig_tokens[:idx_target])

    return corrupted_sentence, decoded_up_to, target_token
    
class Scheme():
    """
    A class for performing text modifications using different schemes.
    This class provides methods to manipulate text through word swapping, word dropping,
    and character-level editing operations. It's designed to work with template-based
    text transformations and tokenization models.
    Attributes:
        source (str): The source text or pattern to be modified or searched for.
        target (str): The target text or pattern to replace the source with.
    Methods:
        swap_words(text): Swaps words in the text based on source and target patterns.
        drop_words(text): Drops specific words from the text and replaces them.
        char_edit(text, model, needed_len, target_id): Performs character-level editing
            operations (swap, drop, add) on tokenized text at a specific position.
    Example:
        scheme = Scheme(source="old_word", target="new_word")
        result = scheme.swap_words("This is old_word in text")
    """
    def __init__(self, source='', target=''):
        self.source = source
        self.target = target
    
    def swap_words(self, text):
        if not template_searcher(seq= text,
                        num_context= 26,
                        num_post_word= 22,
                        source= self.source,
                        target= self.target):
            return None
        
        outputs = text_swap(text= text,
                        text1= self.source,
                        text2= self.target)
        return outputs, (self.source, self.target)

    def drop_words(self, text):
        if not template_searcher(seq= text,
                      num_context= 26,
                      num_post_word= 22,
                      source= self.source,
                        target= self.target):
            return None

        outputs = text_drop(text= text,
                            drop= self.source,
                            replace= self.target)
        return outputs, (self.source, self.target)
        
    def char_edit(self, text, model, needed_len = 47, target_id = 26):
        # tokenize the token
        tokens = model.to_tokens(text, prepend_bos = False)
        if len(tokens[0]) < needed_len:
            return None

        orig_tokens = []
        for tok in tokens[0]:
            orig_tokens.append(model.to_string(tok))

        # select the word id
        target_token = orig_tokens[target_id]
        return_outputs_dict = {}
        for method in ["swap", "drop", "add"]:
            output = get_outputs_modify(orig_tokens, 
                                        target_token,
                                        idx_target= target_id,
                                        method= method)
            if output is None:
                return None
            return_outputs_dict[method] = output

        return return_outputs_dict