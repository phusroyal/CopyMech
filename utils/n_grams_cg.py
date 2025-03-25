import torch
from collections import defaultdict
from jaxtyping import Float
from functools import partial
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint

from .misc import (
    get_top_k,
    get_acc
)

def detect_ngram_copy(seq_ids: torch.Tensor, n=3, skip_up_to=43):
    """
    Minimal function that tries to find n-gram copy scenario
    """
    T = seq_ids.size(1)  # shape (B=1, T)
    if T < n:
        return None, None
    # 1) last token
    last_token = seq_ids[0, -1].item()
    # 2) find earlier positions of last_token
    possible_pos = (seq_ids[0, :-1] == last_token).nonzero().view(-1)
    if possible_pos.numel() == 0:
        return None, None
    # 3) check (n-1) context
    n_minus_1 = n - 1
    context_needed = seq_ids[0, -(n_minus_1+1):-1]  # last n-1 tokens
    matched_pos = None
    for pos in reversed(possible_pos):
        if pos >= n_minus_1:
            candidate = seq_ids[0, pos-n_minus_1:pos]
            if torch.all(candidate == context_needed):
                matched_pos = pos.item()
                break
    if matched_pos is None:
        return None, None
    else:
        return matched_pos, skip_up_to

def ngram_cg(model, skip_up_to, edited_phrases, schema, n=5, k=100, gen_num=20):
    # print("n-gram: ", n)
    # print("Skip layers: ", skip_up_to)

    total_patched_words = 20
    total_solvable_og = 0
    return_scores = []    

    for edited in edited_phrases:
        if total_solvable_og == 100:
            break
        
        outputs = schema(edited)
        if not outputs:
            continue

        # preprocess text
        corrupted_text, pre_isare, correct_tobe = outputs[0]
        source, target = outputs[1]
        prompt = f"Please fix grammar of the following text: '{corrupted_text}'. The correct text is: {pre_isare}"
        prompt_tokens = model.to_tokens(prompt, prepend_bos=False)

        # run on the prompt once with cache to store activations to patch in later
        og_logits, og_cache = model.run_with_cache(prompt_tokens)
        # get the top k tokens
        og_topk_indices = get_top_k(og_logits, k)
        # get the highest prob token
        og_next_token = torch.tensor([og_topk_indices[0]]).unsqueeze(0).to(og_logits.device)

        # check if model can solve the task
        decoded_og_next_token = model.to_string(og_next_token)[0]
        if target in decoded_og_next_token and correct_tobe == target and target != '':
            total_solvable_og += 1
        elif source in decoded_og_next_token and correct_tobe == source:
            total_solvable_og += 1
        else:
            continue
        
        og_topk_lst = []
        # concat next tokens
        og_prompt_tokens = torch.cat([prompt_tokens, og_next_token], dim=1)  
        og_topk_lst.append(og_topk_indices)
        # iterate to gen next gen_num tokens
        for id in range(gen_num):
            logits = model(og_prompt_tokens)
            # get the top k tokens
            og_topk_indices = get_top_k(logits, k)
            # get the highest prob token
            og_next_token = torch.tensor([og_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
            # concat next tokens
            og_prompt_tokens = torch.cat([og_prompt_tokens, og_next_token], dim=1)  
            # store
            og_topk_lst.append(og_topk_indices)
            # og_next_token_lst.append(og_next_token)
        
        patching_succeed_flag = True
        # start on patching model
        score_list_dict = defaultdict(list)
        # score_list = []        
        # pt_topk_lst = []
        # pt_next_token_lst = []
        for num_word2patch in range(total_patched_words, total_patched_words+1):
            if not patching_succeed_flag:
                break

            dict_pred_info = defaultdict(dict)

            pos_matched = []
            pos_current = []
            # total_matches = []

            for id in range(num_word2patch):
                assert id < len(prompt_tokens[0])
                pos_matched.append(detect_ngram_copy(prompt_tokens[:, :len(prompt_tokens[0])-id], n=n)[0])
                pos_current.append(len(prompt_tokens[0])-id-1)
            
            # if there any none of finding ngram, break the experiment with current prompt
            # as it does not have enough tokens
            if None in pos_matched or len(pos_matched) == 0:
                total_solvable_og -= 1
                patching_succeed_flag = False
                break        

            # start hooking
            def residual_stream_patching_hook(
                resid_pre: Float[torch.Tensor, "batch pos d_model"],
                hook: HookPoint,
                pos_matched: list,
                pos_current: list
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                # Each HookPoint has a name attribute giving the name of the hook.
                clean_resid_pre = og_cache[hook.name]
                resid_pre[:, pos_current, :] = clean_resid_pre[:, pos_matched, :]
                return resid_pre
            
            # Use functools.partial to create a temporary hook function with the position fixed
            temp_hook_fn = partial(residual_stream_patching_hook, pos_matched=pos_matched, pos_current=pos_current)

            pt_prompt_tokens = prompt_tokens.clone()
            for idx in range(gen_num+1):
                # Run the model with the patching hook
                patched_logits = model.run_with_hooks(pt_prompt_tokens, fwd_hooks=[
                    (utils.get_act_name("resid_pre", skip_up_to), temp_hook_fn)
                ])
                # get the top k tokens
                tp_topk_indices = get_top_k(patched_logits, k)
                # get the highest prob token
                tp_next_token = torch.tensor([tp_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
                # concat next tokens
                pt_prompt_tokens = torch.cat([pt_prompt_tokens, tp_next_token], dim=1)  

                # pt_topk_indices = get_top_k(patched_logits, k)
                # get the highest prob token
                # pt_next_token = torch.tensor([pt_topk_indices[0]]).unsqueeze(0).to(og_logits.device)

                # # compare the predicted to the og prediction
                # if torch.equal(og_next_token, pt_next_token):
                #     total_matches.append(1)
                # else:
                #     total_matches.append(0)

                dict_pred_info[idx]['original'] = og_topk_lst[idx]
                dict_pred_info[idx]['copy'] = tp_topk_indices
        
            jcc_avg, acc_avg, jcc_all, acc_all = get_acc([dict_pred_info], return_all= True)
            score_list_dict["acc2"].append(acc_avg)
            score_list_dict["jcc"].append(jcc_avg)
            score_list_dict["acc_all"].append(acc_all)
            score_list_dict["jcc_all"].append(jcc_all)
            # print(score_list_dict)
        
        if patching_succeed_flag:
            return_scores.append(score_list_dict)

    return return_scores

def ngram_char_edits_cg(model, skip_up_to, edited_phrases, schema, n=5, k=100, gen_num=20):
    # print("n-gram: ", n)
    # print("Skip layers: ", skip_up_to)

    return_scores = [] 
    total_patched_words = 20
    total_solvable_dict = {
        'swap': 0,
        'drop': 0,
        'add': 0
    }

    for edited in edited_phrases:
        if total_solvable_dict['swap'] == 33 and \
            total_solvable_dict['drop'] == 33 and \
                total_solvable_dict['add'] == 34:
            break
        
        return_outputs = schema(text = edited, model = model)
        if not return_outputs:
            continue

        # preprocess text
        for method, outputs in return_outputs.items():
            if total_solvable_dict[method] == 33 and method in ['swap', 'drop']:
                continue
            if total_solvable_dict[method] == 34 and method in ['add']:
                continue
            corrupted_sentence, decoded_up_to, ground_truth_next = outputs
            prompt = f"Please fix grammar of the following text: '{corrupted_sentence}'. The correct text is: {decoded_up_to}"
            prompt_tokens = model.to_tokens(prompt, prepend_bos=False)

            # As we does not need on the turning point, we can skip the last tokens
            # Prompt: A B C D is .... A B C [D]
            # Get the last token of the prompt as the token to predict
            next_token_ref = prompt_tokens[:, -1]
            prompt_tokens = prompt_tokens[:, :-1] 

            # run on the prompt once with cache to store activations to patch in later
            og_logits, og_cache = model.run_with_cache(prompt_tokens)
            # get the top k tokens
            og_topk_indices = get_top_k(og_logits, k)
            # get the highest prob token
            og_next_token = torch.tensor([og_topk_indices[0]]).unsqueeze(0).to(og_logits.device)

            # check if model can solve the task (now, D should be predicted, aka next_token_ref)
            if torch.equal(og_next_token[0], next_token_ref):
                total_solvable_dict[method] += 1
            else:
                continue

            og_topk_lst = []
            # concat next tokens
            og_prompt_tokens = torch.cat([prompt_tokens, og_next_token], dim=1)  
            og_topk_lst.append(og_topk_indices)
            # iterate to gen next gen_num tokens
            for id in range(gen_num):
                logits = model(og_prompt_tokens)
                # get the top k tokens
                og_topk_indices = get_top_k(logits, k)
                # get the highest prob token
                og_next_token = torch.tensor([og_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
                # concat next tokens
                og_prompt_tokens = torch.cat([og_prompt_tokens, og_next_token], dim=1)  
                # store
                og_topk_lst.append(og_topk_indices)
                # og_next_token_lst.append(og_next_token)

            patching_succeed_flag = True
            score_list_dict = {
                'acc2': [],
                'acc3': [],
                'jcc': []
            }       
            # start on patching model
            for num_word2patch in range(total_patched_words, total_patched_words+1):
                if not patching_succeed_flag:
                    break

                dict_pred_info = defaultdict(dict)

                pos_matched = []
                pos_current = []
                total_matches = []

                for id in range(num_word2patch):
                    assert id < len(prompt_tokens[0])
                    pos_matched.append(detect_ngram_copy(prompt_tokens[:, :len(prompt_tokens[0])-id], n=n)[0])
                    pos_current.append(len(prompt_tokens[0])-id-1)

                # if there any none of finding ngram, break the experiment with current prompt
                # as it does not have enough tokens
                if None in pos_matched:
                    total_solvable_dict[method] -= 1
                    patching_succeed_flag = False
                    break            
                    
                # start hooking
                def residual_stream_patching_hook(
                    resid_pre: Float[torch.Tensor, "batch pos d_model"],
                    hook: HookPoint,
                    pos_matched: list,
                    pos_current: list
                ) -> Float[torch.Tensor, "batch pos d_model"]:
                    # Each HookPoint has a name attribute giving the name of the hook.
                    clean_resid_pre = og_cache[hook.name]
                    resid_pre[:, pos_current, :] = clean_resid_pre[:, pos_matched, :]
                    return resid_pre
                
                # Use functools.partial to create a temporary hook function with the position fixed
                temp_hook_fn = partial(residual_stream_patching_hook, pos_matched=pos_matched, pos_current=pos_current)
                
                pt_prompt_tokens = prompt_tokens.clone()
                for idx in range(gen_num+1):
                    # Run the model with the patching hook
                    patched_logits = model.run_with_hooks(pt_prompt_tokens, fwd_hooks=[
                        (utils.get_act_name("resid_pre", skip_up_to), temp_hook_fn)
                    ])
                    # get the top k tokens
                    tp_topk_indices = get_top_k(patched_logits, k)
                    # get the highest prob token
                    tp_next_token = torch.tensor([tp_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
                    # concat next tokens
                    pt_prompt_tokens = torch.cat([pt_prompt_tokens, tp_next_token], dim=1)
                

                    dict_pred_info[idx]['original'] = og_topk_lst[idx]
                    dict_pred_info[idx]['copy'] = tp_topk_indices
            
                jcc_avg, acc_avg, jcc_all, acc_all = get_acc([dict_pred_info], return_all= True)
                score_list_dict["acc2"].append(acc_avg)
                score_list_dict["jcc"].append(jcc_avg)
                score_list_dict["acc_all"].append(acc_all)
                score_list_dict["jcc_all"].append(jcc_all)

            if patching_succeed_flag:
                return_scores.append(score_list_dict)
    
    return return_scores

# def residual_stream_patching_hook(
#     resid_pre: Float[torch.Tensor, "batch pos d_model"],
#     hook: HookPoint,
#     pos_matched: list,
#     pos_current: list
# ) -> Float[torch.Tensor, "batch pos d_model"]:
#     # Each HookPoint has a name attribute giving the name of the hook.
#     clean_resid_pre = og_cache[hook.name]
#     resid_pre[:, pos_current, :] = clean_resid_pre[:, pos_matched, :]
#     return resid_pre


# # Use functools.partial to create a temporary hook function with the position fixed
# temp_hook_fn = partial(residual_stream_patching_hook, pos_matched=pos_matched, pos_current=pos_current)
# # Run the model with the patching hook
# patched_logits = model.run_with_hooks(prompt_tokens, fwd_hooks=[
#     (utils.get_act_name("resid_pre", skip_up_to), temp_hook_fn)
# ])

# def v_patching_hook(
#     resid_pre: Float[torch.Tensor, "batch pos head_index d_head"],
#     hook: HookPoint,
#     position: int
# ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
#     # Each HookPoint has a name attribute giving the name of the hook.
#     clean_resid_pre = og_cache[hook.name]
#     resid_pre[:, -1, :, :] = clean_resid_pre[:, position, :, :]
#     return resid_pre

# for layer in range(skip_up_to):
#     # Use functools.partial to create a temporary hook function with the position fixed
#     temp_hook_fn = partial(v_patching_hook, position=t_matched)
#     # Run the model with the patching hook
#     patched_logits = model.run_with_hooks(prompt_tokens, fwd_hooks=[
#         (utils.get_act_name("v", layer), temp_hook_fn)
#     ])