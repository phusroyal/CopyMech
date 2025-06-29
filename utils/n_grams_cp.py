import torch
from collections import defaultdict
from jaxtyping import Float
from functools import partial
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint

from .misc import get_top_k, get_acc, detect_ngram_copy

def ngram_cp(model, skip_up_to, edited_phrases, schema, n=5, k=100, solvable_limit=100):
    """
    Run copy mode schema for a given model and edited phrases.

    Args:
        model: TransformerLens model.
        skip_up_to (int): Layer to patch.
        edited_phrases (list): List of phrase data for the experiment.
        schema (callable): Function that extracts (corrupted, pre, correct) tuple from a phrase.
        n (int): N-gram size.
        k (int): Number of top-k tokens to evaluate.

    Returns:
        list: Each entry is a dict of patch-copying scores (accuracy, Jaccard) for an input.
    """
    total_patched_words = 20
    total_solvable_og = 0
    return_scores = []

    for edited in edited_phrases:
        if total_solvable_og == solvable_limit:
            break
        outputs = schema(edited)
        if not outputs:
            continue

        # Unpack the input 
        corrupted_text, pre_isare, correct_tobe = outputs[0]
        prompt = f"Please fix grammar of the following text: '{corrupted_text}'. The correct text is: {pre_isare}"
        
        # Prompt: A B C D is .... A B C D [are]
        prompt_tokens = model.to_tokens(prompt, prepend_bos=False)

        # As we does not need on the turning point, we can skip the last tokens
        # Prompt: A B C D is .... A B C [D]
        # Get the last token of the prompt as the token to predict
        next_token_ref = prompt_tokens[:, -1]
        prompt_tokens = prompt_tokens[:, :-1]

        # Run the model to get original logits and cache
        og_logits, og_cache = model.run_with_cache(prompt_tokens)
        og_topk_indices = get_top_k(og_logits, k)
        og_next_token = torch.tensor([og_topk_indices[0]]).unsqueeze(0).to(og_logits.device)

        # Only continue if model can predict the correct token at the turning point
        if torch.equal(og_next_token[0], next_token_ref):
            total_solvable_og += 1
        else:
            continue

        patching_succeed_flag = True
        score_list_dict = {'acc2': [], 'acc3': [], 'jcc': []}

        # Loop through the number of words to patch
        # We will patch up to 20 words, but can stop earlier if patching fails
        # We will patch the last word first, then the second last, and so on
        for num_word2patch in range(1, total_patched_words+1):
            # If patching has already failed, break the loop
            if not patching_succeed_flag:
                break

            # Initialize a dictionary to store predictions
            dict_pred_info = defaultdict(dict)
            pos_matched = []
            pos_current = []
            total_matches = []

            # For each word to patch, find the position of the n-gram copy
            for id in range(num_word2patch):
                assert id < len(prompt_tokens[0])
                match_idx, _ = detect_ngram_copy(prompt_tokens[:, :len(prompt_tokens[0])-id], n=n)
                pos_matched.append(match_idx)
                pos_current.append(len(prompt_tokens[0])-id-1)

            # If any position is None, patching fails
            # This means we cannot find a valid n-gram copy for the current word
            # or we have no words to patch
            if None in pos_matched or len(pos_matched) == 0:
                total_solvable_og -= 1
                patching_succeed_flag = False
                break

            # Define hook for patch-copying activations
            def residual_stream_patching_hook(
                resid_pre: Float[torch.Tensor, "batch pos d_model"],
                hook: HookPoint,
                pos_matched: list,
                pos_current: list
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                clean_resid_pre = og_cache[hook.name]
                resid_pre[:, pos_current, :] = clean_resid_pre[:, pos_matched, :]
                return resid_pre

            # Create a temporary hook function with the matched positions
            temp_hook_fn = partial(residual_stream_patching_hook, pos_matched=pos_matched, pos_current=pos_current)

            # Run the model with the patching hook
            patched_logits = model.run_with_hooks(prompt_tokens, fwd_hooks=[
                (utils.get_act_name("resid_pre", skip_up_to), temp_hook_fn)
            ])
            pt_topk_indices = get_top_k(patched_logits, k)
            pt_next_token = torch.tensor([pt_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
            
            # Check if the patched next token matches the reference next token
            total_matches.append(int(torch.equal(next_token_ref, pt_next_token[0])))
            
            # Store the original and patched top-k indices in the prediction info
            # acc2, acc3 are just the accuracy of the top-k predictions 
            # but in different ways
            dict_pred_info[0]['original'] = og_topk_indices
            dict_pred_info[0]['copy'] = get_top_k(patched_logits, k)
            jcc, acc = get_acc([dict_pred_info])
            score_list_dict["acc2"].append(acc)
            score_list_dict["jcc"].append(jcc)
            score_list_dict["acc3"].append(sum(total_matches) / len(total_matches))

        # If patching was successful, append the scores to the return list
        if patching_succeed_flag:
            return_scores.append(score_list_dict)
    
    print(f"Total solvable examples: {total_solvable_og}")
    return return_scores


def ngram_char_edits_cp(model, skip_up_to, edited_phrases, schema, n=5, k=100, solvable_limit=33):
    """
    Run copy mode schema for a given model and edited phrases with character-level edits.
    
    Args:
        model: TransformerLens model.
        skip_up_to (int): Layer to patch.
        edited_phrases (list): List of phrase data for the experiment.
        schema (callable): Returns a dict for each edit type mapping to (corrupted, up_to, ground_truth_next).
        n (int): N-gram size.
        k (int): Number of top-k tokens to evaluate.

    Returns:
        list: Each entry is a dict of patch-copying scores for an input.
    """
    return_scores = []
    total_patched_words = 20
    total_solvable_dict = {'swap': 0, 'drop': 0, 'add': 0}

    for edited in edited_phrases:
        # Stop if we have enough examples for all three edit types
        if all([
            total_solvable_dict['swap'] == solvable_limit,
            total_solvable_dict['drop'] == solvable_limit,
            total_solvable_dict['add'] == solvable_limit+1, 
        ]):
            break
        return_outputs = schema(text=edited, model=model)
        if not return_outputs:
            continue

        # Unpack the outputs for each method
        for method, outputs in return_outputs.items():
            # Limit number of samples for each method
            if method in ['swap', 'drop'] and total_solvable_dict[method] == 33:
                continue
            if method == 'add' and total_solvable_dict[method] == 34:
                continue

            # Unpack the outputs
            corrupted_sentence, decoded_up_to, ground_truth_next = outputs
            prompt = f"Please fix grammar of the following text: '{corrupted_sentence}'. The correct text is: {decoded_up_to}"
            prompt_tokens = model.to_tokens(prompt, prepend_bos=False)

            # As we does not need on the turning point, we can skip the last tokens
            # Prompt: A B C D is .... A B C [D]
            # Get the last token of the prompt as the token to predict
            next_token_ref = prompt_tokens[:, -1]
            prompt_tokens = prompt_tokens[:, :-1]

            # Run the model to get original logits and cache
            og_logits, og_cache = model.run_with_cache(prompt_tokens)
            og_topk_indices = get_top_k(og_logits, k)
            og_next_token = torch.tensor([og_topk_indices[0]]).unsqueeze(0).to(og_logits.device)

            # Only continue if model can predict the correct token at the turning point
            if torch.equal(og_next_token[0], next_token_ref):
                total_solvable_dict[method] += 1
            else:
                continue


            patching_succeed_flag = True
            score_list_dict = {'acc2': [], 'acc3': [], 'jcc': []}

            # Loop through the number of words to patch
            for num_word2patch in range(1, total_patched_words+1):
                if not patching_succeed_flag:
                    break

                # Initialize a dictionary to store predictions
                dict_pred_info = defaultdict(dict)
                pos_matched = []
                pos_current = []
                total_matches = []

                # For each word to patch, find the position of the n-gram copy
                for id in range(num_word2patch):
                    assert id < len(prompt_tokens[0])
                    match_idx, _ = detect_ngram_copy(prompt_tokens[:, :len(prompt_tokens[0])-id], n=n)
                    pos_matched.append(match_idx)
                    pos_current.append(len(prompt_tokens[0])-id-1)

                # If any position is None, patching fails
                # This means we cannot find a valid n-gram copy for the current word
                # or we have no words to patch
                if None in pos_matched:
                    total_solvable_dict[method] -= 1
                    patching_succeed_flag = False
                    break

                # Define hook for patch-copying activations
                def residual_stream_patching_hook(
                    resid_pre: Float[torch.Tensor, "batch pos d_model"],
                    hook: HookPoint,
                    pos_matched: list,
                    pos_current: list
                ) -> Float[torch.Tensor, "batch pos d_model"]:
                    clean_resid_pre = og_cache[hook.name]
                    resid_pre[:, pos_current, :] = clean_resid_pre[:, pos_matched, :]
                    return resid_pre

                # Create a temporary hook function with the matched positions
                temp_hook_fn = partial(residual_stream_patching_hook, pos_matched=pos_matched, pos_current=pos_current)
                
                # Run the model with the patching hook
                patched_logits = model.run_with_hooks(prompt_tokens, fwd_hooks=[
                    (utils.get_act_name("resid_pre", skip_up_to), temp_hook_fn)
                ])
                pt_topk_indices = get_top_k(patched_logits, k)
                pt_next_token = torch.tensor([pt_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
                
                # Check if the patched next token matches the reference next token
                total_matches.append(int(torch.equal(next_token_ref, pt_next_token[0])))

                # Store the original and patched top-k indices in the prediction info
                # acc2, acc3 are just the accuracy of the top-k predictions
                # but in different ways
                dict_pred_info[0]['original'] = og_topk_indices
                dict_pred_info[0]['copy'] = get_top_k(patched_logits, k)
                jcc, acc = get_acc([dict_pred_info])
                score_list_dict["acc2"].append(acc)
                score_list_dict["jcc"].append(jcc)
                score_list_dict["acc3"].append(sum(total_matches) / len(total_matches))

            # If patching was successful, append the scores to the return list
            if patching_succeed_flag:
                return_scores.append(score_list_dict)
    
    print(f"Total solvable examples: {total_solvable_dict}")
    return return_scores
