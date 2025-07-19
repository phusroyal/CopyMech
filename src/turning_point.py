import torch
from collections import defaultdict
from jaxtyping import Float
from functools import partial
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint

from .misc import (
    get_top_k,
    get_acc,
    detect_ngram_copy,
)

def ngram(model, skip_up_to, edited_phrases, schema, n=5, k=100):
    """
    Run turning point schema for a given model and edited phrases.

    Args:
        model: TransformerLens model.
        skip_up_to (int): Layer to apply the patch-copying intervention.
        edited_phrases (list): Dataset, each item produces a prompt via `schema`.
        schema (callable): Function that returns (corrupted_text, pre_isare, correct_tobe), (source, target).
        n (int): N-gram window size for copy detection.
        k (int): Top-k for logits evaluation.

    Returns:
        List[Dict]: Each dict contains accuracy and Jaccard metrics for one input.
    """
    total_patched_words = 20
    total_solvable_og = 0
    return_scores = []
    solvable_cases = []

    for edited in edited_phrases:
        if total_solvable_og == 100:
            break

        outputs = schema(edited)
        if not outputs:
            continue

        # Unpack the prompt fields
        corrupted_text, pre_isare, correct_tobe = outputs[0]
        source, target = outputs[1]
        prompt = f"Please fix grammar of the following text: '{corrupted_text}'. The correct text is: {pre_isare}"
        prompt_tokens = model.to_tokens(prompt, prepend_bos=False)

        # Run with cache to store residual stream activations
        og_logits, og_cache = model.run_with_cache(prompt_tokens)
        og_topk_indices = get_top_k(og_logits, k)
        og_next_token = torch.tensor([og_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
        decoded_og_next_token = model.to_string(og_next_token)[0]

        # Only proceed if model produces the correct target
        if (target in decoded_og_next_token and correct_tobe == target and target != '') or \
           (source in decoded_og_next_token and correct_tobe == source):
            total_solvable_og += 1
        else:
            continue

        patching_succeed_flag = True
        score_list_dict = {'acc2': [], 'acc3': [], 'jcc': []}

        for num_word2patch in range(1, total_patched_words+1):
            if not patching_succeed_flag:
                break

            dict_pred_info = defaultdict(dict)
            pos_matched = []
            pos_current = []
            total_matches = []

            # Compute which positions to patch/copy
            for id in range(num_word2patch):
                assert id < len(prompt_tokens[0])
                match_idx, _ = detect_ngram_copy(prompt_tokens[:, :len(prompt_tokens[0])-id], n=n)
                pos_matched.append(match_idx)
                pos_current.append(len(prompt_tokens[0])-id-1)

            # If no valid positions found, skip patching
            # or if all positions are None, skip patching
            if None in pos_matched or len(pos_matched) == 0:
                total_solvable_og -= 1
                patching_succeed_flag = False
                break

            # Define patching hook for the residual stream
            def residual_stream_patching_hook(
                resid_pre: Float[torch.Tensor, "batch pos d_model"],
                hook: HookPoint,
                pos_matched: list,
                pos_current: list
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                clean_resid_pre = og_cache[hook.name]
                resid_pre[:, pos_current, :] = clean_resid_pre[:, pos_matched, :]
                return resid_pre
            # Apply the patching hook
            temp_hook_fn = partial(residual_stream_patching_hook, pos_matched=pos_matched, pos_current=pos_current)
            
            # Run the model with the patching hook
            patched_logits = model.run_with_hooks(prompt_tokens, fwd_hooks=[
                (utils.get_act_name("resid_pre", skip_up_to), temp_hook_fn)
            ])
            pt_topk_indices = get_top_k(patched_logits, k)
            pt_next_token = torch.tensor([pt_topk_indices[0]]).unsqueeze(0).to(og_logits.device)

            # Compare if patched and original prediction match
            if torch.equal(og_next_token, pt_next_token):
                total_matches.append(1)
            else:
                total_matches.append(0)

            # Store prediction information
            dict_pred_info[0]['original'] = og_topk_indices
            dict_pred_info[0]['copy'] = pt_topk_indices

            jcc, acc = get_acc([dict_pred_info])
            score_list_dict["acc2"].append(acc)
            score_list_dict["jcc"].append(jcc)
            score_list_dict["acc3"].append(sum(total_matches) / len(total_matches))

        if patching_succeed_flag:
            return_scores.append(score_list_dict)
            solvable_cases.append({
                'corrupted_text': corrupted_text,
                'pre_isare': pre_isare,
                'correct_tobe': correct_tobe,
                'source': source,
                'target': target,
            })

    return return_scores, solvable_cases

def ngram_char_edits(model, skip_up_to, edited_phrases, schema, n=5, k=100):
    """
    Run turning point schema for a given model and edited phrases with character-level edits.

    Args:
        model: TransformerLens model.
        skip_up_to (int): Layer to apply the patch.
        edited_phrases (list): List of data items for edit evaluation.
        schema (callable): Function returning {edit_type: (corrupted_sentence, decoded_up_to, ground_truth_next)}.
        n (int): N-gram window size for copy detection.
        k (int): Top-k for logits evaluation.

    Returns:
        List[Dict]: Each dict contains accuracy and Jaccard metrics for an input.
    """
    return_scores = []
    solvable_cases = []
    total_patched_words = 20
    total_solvable_dict = {'swap': 0, 'drop': 0, 'add': 0}

    for edited in edited_phrases:
        # Stop if all edit types are solved
        if (total_solvable_dict['swap'] == 33 and 
            total_solvable_dict['drop'] == 33 and 
            total_solvable_dict['add'] == 34):
            break

        # Get outputs from the schema
        return_outputs = schema(text=edited, model=model)
        if not return_outputs:
            continue

        for method, outputs in return_outputs.items():
            # Limit samples per edit type
            if method in ['swap', 'drop'] and total_solvable_dict[method] == 33:
                continue
            if method == 'add' and total_solvable_dict[method] == 34:
                continue

            # Unpack the outputs
            corrupted_sentence, decoded_up_to, ground_truth_next = outputs
            prompt = f"Please fix grammar of the following text: '{corrupted_sentence}'. The correct text is: {decoded_up_to}"
            prompt_tokens = model.to_tokens(prompt, prepend_bos=False)

            # Run the model with cache to store activations
            og_logits, og_cache = model.run_with_cache(prompt_tokens)
            og_topk_indices = get_top_k(og_logits, k)
            og_next_token = torch.tensor([og_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
            decoded_og_next_token = model.to_string(og_next_token)[0]

            # Check if the model can solve the task
            if ground_truth_next in decoded_og_next_token:
                total_solvable_dict[method] += 1
            else:
                continue

            patching_succeed_flag = True
            score_list_dict = {'acc2': [], 'acc3': [], 'jcc': []}

            # Start patching the model
            for num_word2patch in range(1, total_patched_words+1):
                if not patching_succeed_flag:
                    break

                dict_pred_info = defaultdict(dict)
                pos_matched = []
                pos_current = []
                total_matches = []

                # Compute positions to patch/copy
                for id in range(num_word2patch):
                    assert id < len(prompt_tokens[0])
                    match_idx, _ = detect_ngram_copy(prompt_tokens[:, :len(prompt_tokens[0])-id], n=n)
                    pos_matched.append(match_idx)
                    pos_current.append(len(prompt_tokens[0])-id-1)

                # If no valid positions found, skip patching
                if None in pos_matched:
                    total_solvable_dict[method] -= 1
                    patching_succeed_flag = False
                    break

                # Define patching hook for the residual stream
                def residual_stream_patching_hook(
                    resid_pre: Float[torch.Tensor, "batch pos d_model"],
                    hook: HookPoint,
                    pos_matched: list,
                    pos_current: list
                ) -> Float[torch.Tensor, "batch pos d_model"]:
                    clean_resid_pre = og_cache[hook.name]
                    resid_pre[:, pos_current, :] = clean_resid_pre[:, pos_matched, :]
                    return resid_pre
                # Apply the patching hook
                temp_hook_fn = partial(residual_stream_patching_hook, pos_matched=pos_matched, pos_current=pos_current)
                patched_logits = model.run_with_hooks(prompt_tokens, fwd_hooks=[
                    (utils.get_act_name("resid_pre", skip_up_to), temp_hook_fn)
                ])
                pt_topk_indices = get_top_k(patched_logits, k)
                pt_next_token = torch.tensor([pt_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
                
                # Compare if patched and original prediction match
                if torch.equal(og_next_token, pt_next_token):
                    total_matches.append(1)
                else:
                    total_matches.append(0)

                # Store prediction information
                dict_pred_info[0]['original'] = og_topk_indices
                dict_pred_info[0]['copy'] = pt_topk_indices

                jcc, acc = get_acc([dict_pred_info])
                score_list_dict["acc2"].append(acc)
                score_list_dict["jcc"].append(jcc)
                score_list_dict["acc3"].append(sum(total_matches) / len(total_matches))

            # If patching was successful, store the scores
            if patching_succeed_flag:
                return_scores.append(score_list_dict)
                solvable_cases.append({
                    'corrupted_sentence': corrupted_sentence,
                    'decoded_up_to': decoded_up_to,
                    'ground_truth_next': ground_truth_next,
                    'method': method
                })

    return return_scores, solvable_cases
