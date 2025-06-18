import torch
from collections import defaultdict
from jaxtyping import Float
from functools import partial
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint

from .misc import (
    get_top_k,
    get_acc,
    compute_bleu,
    compute_rouge_l,
    detect_ngram_copy
)
def detect_extra_token(list1, list2):
    """
    Detect and remove a single extra token at the end of either list,
    if present, based on matching patterns at the end of the lists.

    Args:
        list1 (list): First token list.
        list2 (list): Second token list.

    Returns:
        tuple: Possibly truncated (list1, list2) after removing the extra token.
    """
    if len(list1) < 2 or len(list2) < 2:
        return list1, list2

    # Compare last token of list1 with second-to-last of list2
    if list1[-1] == list2[-2]:
        return list1, list2[:-1]
    # Compare last token of list2 with second-to-last of list1
    elif list2[-1] == list1[-2]:
        return list1[:-1], list2
    else:
        return list1, list2

def ngram_cg(
    model,
    skip_up_to,
    edited_phrases,
    schema,
    n=5,
    k=100,
    gen_num=20
):
    """
    Run continuation generation schema for a given model and edited phrases.

    Args:
        model: TransformerLens model instance.
        skip_up_to (int): Layer at which to apply the patch.
        edited_phrases (list): List of edited phrase examples.
        schema (callable): Function for extracting text fields from data.
        n (int): N-gram size for copy detection.
        k (int): Top-k tokens to consider for generation.
        gen_num (int): Number of tokens to autoregressively generate.

    Returns:
        list: A list of dictionaries containing metrics for each example.
    """
    total_patched_words = 20
    total_solvable_og = 0
    return_scores = []

    for edited in edited_phrases:
        if total_solvable_og == 100:
            break

        outputs = schema(edited)
        if not outputs:
            continue

        # Unpack the input
        corrupted_text, pre_isare, correct_tobe = outputs[0]
        source, target = outputs[1]
        prompt = f"Please fix grammar of the following text: '{corrupted_text}'. The correct text is: {pre_isare}"
        prompt_tokens = model.to_tokens(prompt, prepend_bos=False)

        # Run once to cache activations for later patching
        og_logits, og_cache = model.run_with_cache(prompt_tokens)
        og_topk_indices = get_top_k(og_logits, k)
        og_next_token = torch.tensor([og_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
        decoded_og_next_token = model.to_string(og_next_token)[0]

        # Check if the model solves the task
        if (target in decoded_og_next_token and correct_tobe == target and target != '') or \
           (source in decoded_og_next_token and correct_tobe == source):
            total_solvable_og += 1
        else:
            continue

        # Prepare for patch-copying
        og_topk_lst = []
        og_prompt_tokens = torch.cat([prompt_tokens, og_next_token], dim=1)
        og_topk_lst.append(og_topk_indices)
        for _ in range(gen_num):
            logits = model(og_prompt_tokens)
            og_topk_indices = get_top_k(logits, k)
            og_next_token = torch.tensor([og_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
            og_prompt_tokens = torch.cat([og_prompt_tokens, og_next_token], dim=1)
            og_topk_lst.append(og_topk_indices)

        # Initialize patching variables
        patching_succeed_flag = True
        score_list_dict = defaultdict(list)
        prompt_num_tokens = prompt_tokens.shape[1]

        # Perform patch-copying for a fixed number of words
        for num_word2patch in range(total_patched_words, total_patched_words+1):
            if not patching_succeed_flag:
                break

            dict_pred_info = defaultdict(dict)
            pos_matched = []
            pos_current = []

            # Detect n-gram copies in the prompt tokens
            for idx in range(num_word2patch):
                assert idx < len(prompt_tokens[0])
                matched_pos, _ = detect_ngram_copy(prompt_tokens[:, :len(prompt_tokens[0])-idx], n=n)
                pos_matched.append(matched_pos)
                pos_current.append(len(prompt_tokens[0])-idx-1)

            # If no valid positions found, skip patching
            if None in pos_matched or len(pos_matched) == 0:
                total_solvable_og -= 1
                patching_succeed_flag = False
                break

            # Define the patching hook
            def residual_stream_patching_hook(
                resid_pre: Float[torch.Tensor, "batch pos d_model"],
                hook: HookPoint,
                pos_matched: list,
                pos_current: list
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                clean_resid_pre = og_cache[hook.name]
                resid_pre[:, pos_current, :] = clean_resid_pre[:, pos_matched, :]
                return resid_pre

            # Create a partial function for the hook
            temp_hook_fn = partial(residual_stream_patching_hook, pos_matched=pos_matched, pos_current=pos_current)
            
            # Initialize the patched prompt tokens
            pt_prompt_tokens = prompt_tokens.clone()
            # Generate tokens using the patched prompt
            for idx in range(gen_num+1):
                patched_logits = model.run_with_hooks(pt_prompt_tokens, fwd_hooks=[
                    (utils.get_act_name("resid_pre", skip_up_to), temp_hook_fn)
                ])
                tp_topk_indices = get_top_k(patched_logits, k)
                tp_next_token = torch.tensor([tp_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
                pt_prompt_tokens = torch.cat([pt_prompt_tokens, tp_next_token], dim=1)

                dict_pred_info[idx]['original'] = og_topk_lst[idx]
                dict_pred_info[idx]['copy'] = tp_topk_indices

            # Calculate accuracy and JCC
            jcc_avg, acc_avg, jcc_all, acc_all = get_acc([dict_pred_info], return_all=True)
            score_list_dict["acc2"].append(acc_avg)
            score_list_dict["jcc"].append(jcc_avg)
            score_list_dict["acc_all"].append(acc_all)
            score_list_dict["jcc_all"].append(jcc_all)

            # Compute BLEU and ROUGE-L scores
            # and store generated sequences
            # Note: The prompt tokens are used to generate the text
            # after the initial prompt, so we slice accordingly
            candidate = model.to_string(og_prompt_tokens[0, prompt_num_tokens:]).split()
            reference = model.to_string(pt_prompt_tokens[0, prompt_num_tokens:]).split()
            candidate, reference = detect_extra_token(candidate, reference)
            bleu = compute_bleu(candidate, reference)
            rouge_l = compute_rouge_l(candidate, reference)
            score_list_dict['bleu_rougel'].append([bleu, rouge_l])
            score_list_dict['og_generated'].append(candidate)
            score_list_dict['pt_generated'].append(reference)

        # If patching was successful, store the results
        if patching_succeed_flag:
            return_scores.append(score_list_dict)

    return return_scores

def ngram_char_edits_cg(
    model, 
    skip_up_to, 
    edited_phrases, 
    schema, 
    n=5, 
    k=100, 
    gen_num=20
):
    """
    Run n-gram patch-copying experiments on character-level edit tasks
    (swap, drop, add) for a given model and edited phrases.

    Args:
        model: TransformerLens model instance.
        skip_up_to (int): Layer at which to apply the patch.
        edited_phrases (list): List of edited phrase examples.
        schema (callable): Function for extracting fields for each edit type.
        n (int): N-gram size for copy detection.
        k (int): Top-k tokens to consider for generation.
        gen_num (int): Number of tokens to autoregressively generate.

    Returns:
        list: A list of dictionaries containing metrics for each example.
    """
    return_scores = []
    total_patched_words = 20
    total_solvable_dict = {'swap': 0, 'drop': 0, 'add': 0}

    for edited in edited_phrases:
        # Stop when we have enough samples for all edit types
        if all([
            total_solvable_dict['swap'] == 33,
            total_solvable_dict['drop'] == 33,
            total_solvable_dict['add'] == 34
        ]):
            break

        return_outputs = schema(text=edited, model=model)
        if not return_outputs:
            continue

        # Unpack the outputs
        # return_outputs is a dictionary with keys 'swap', 'drop', 'add'
        # Each key maps to a tuple (corrupted_sentence, decoded_up_to, ground_truth_next)
        # where:
        # - corrupted_sentence: The input text with some corruption.
        # - decoded_up_to: The text that the model has decoded up to this point.
        # - ground_truth_next: The expected next token or text to be generated.
        # We will iterate over each method (swap, drop, add) and process accordingly.
        for method, outputs in return_outputs.items():
            if (method in ['swap', 'drop'] and total_solvable_dict[method] == 33) or \
               (method == 'add' and total_solvable_dict[method] == 34):
                continue

            corrupted_sentence, decoded_up_to, ground_truth_next = outputs
            prompt = f"Please fix grammar of the following text: '{corrupted_sentence}'. The correct text is: {decoded_up_to}"
            prompt_tokens = model.to_tokens(prompt, prepend_bos=False)

            # Run once to cache activations for later patching
            og_logits, og_cache = model.run_with_cache(prompt_tokens)
            og_topk_indices = get_top_k(og_logits, k)
            og_next_token = torch.tensor([og_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
            decoded_og_next_token = model.to_string(og_next_token)[0]

            # Check if the model solves the task
            if ground_truth_next in decoded_og_next_token:
                total_solvable_dict[method] += 1
            else:
                continue

            # Prepare for patch-copying
            og_topk_lst = []
            og_prompt_tokens = torch.cat([prompt_tokens, og_next_token], dim=1)
            og_topk_lst.append(og_topk_indices)
            for _ in range(gen_num):
                logits = model(og_prompt_tokens)
                og_topk_indices = get_top_k(logits, k)
                og_next_token = torch.tensor([og_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
                og_prompt_tokens = torch.cat([og_prompt_tokens, og_next_token], dim=1)
                og_topk_lst.append(og_topk_indices)

            # Initialize patching variables
            patching_succeed_flag = True
            score_list_dict = defaultdict(list)
            prompt_num_tokens = prompt_tokens.shape[1]

            # Perform patch-copying for a fixed number of words
            for num_word2patch in range(total_patched_words, total_patched_words+1):
                if not patching_succeed_flag:
                    break

                # Detect n-gram copies in the prompt tokens
                dict_pred_info = defaultdict(dict)
                pos_matched = []
                pos_current = []

                # We will try to find n-gram copies in the last num_word2patch tokens
                pos_matched = []
                for idx in range(num_word2patch):
                    assert idx < len(prompt_tokens[0])
                    matched_pos, _ = detect_ngram_copy(prompt_tokens[:, :len(prompt_tokens[0])-idx], n=n)
                    pos_matched.append(matched_pos)
                    pos_current.append(len(prompt_tokens[0])-idx-1)

                # If we cannot find valid positions, we will skip the patching
                if None in pos_matched:
                    total_solvable_dict[method] -= 1
                    patching_succeed_flag = False
                    break

                # Define the patching hook
                def residual_stream_patching_hook(
                    resid_pre: Float[torch.Tensor, "batch pos d_model"],
                    hook: HookPoint,
                    pos_matched: list,
                    pos_current: list
                ) -> Float[torch.Tensor, "batch pos d_model"]:
                    clean_resid_pre = og_cache[hook.name]
                    resid_pre[:, pos_current, :] = clean_resid_pre[:, pos_matched, :]
                    return resid_pre
                # Create a partial function for the hook
                temp_hook_fn = partial(residual_stream_patching_hook, pos_matched=pos_matched, pos_current=pos_current)

                # Initialize the patched prompt tokens
                pt_prompt_tokens = prompt_tokens.clone()
                for idx in range(gen_num+1):
                    patched_logits = model.run_with_hooks(pt_prompt_tokens, fwd_hooks=[
                        (utils.get_act_name("resid_pre", skip_up_to), temp_hook_fn)
                    ])
                    tp_topk_indices = get_top_k(patched_logits, k)
                    tp_next_token = torch.tensor([tp_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
                    pt_prompt_tokens = torch.cat([pt_prompt_tokens, tp_next_token], dim=1)

                    dict_pred_info[idx]['original'] = og_topk_lst[idx]
                    dict_pred_info[idx]['copy'] = tp_topk_indices

                # Calculate accuracy and JCC
                jcc_avg, acc_avg, jcc_all, acc_all = get_acc([dict_pred_info], return_all=True)
                score_list_dict["acc2"].append(acc_avg)
                score_list_dict["jcc"].append(jcc_avg)
                score_list_dict["acc_all"].append(acc_all)
                score_list_dict["jcc_all"].append(jcc_all)

                # Compute BLEU and ROUGE-L scores
                # and store generated sequences
                candidate = model.to_string(og_prompt_tokens[0, prompt_num_tokens:]).split()
                reference = model.to_string(pt_prompt_tokens[0, prompt_num_tokens:]).split()
                candidate, reference = detect_extra_token(candidate, reference)
                bleu = compute_bleu(candidate, reference)
                rouge_l = compute_rouge_l(candidate, reference)
                score_list_dict['bleu_rougel'].append([bleu, rouge_l])
                score_list_dict['og_generated'].append(candidate)
                score_list_dict['pt_generated'].append(reference)

            if patching_succeed_flag:
                return_scores.append(score_list_dict)

    return return_scores
