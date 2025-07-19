import torch
from collections import defaultdict
from jaxtyping import Float
from functools import partial
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm

from ..utils.misc import (
    get_top_k,
    get_acc,
    compute_bleu,
    compute_rouge_l,
    detect_ngram_copy
)
from data_loader import cnndnn_loader

def extractive_prompt(article: str) -> str:
    """
    Generate a prompt for extractive summarization.
    ref: https://www.alphaxiv.org/abs/2502.08923
    
    Args:
        article: The article text to summarize.
    
    Returns:
        A formatted prompt string.
    """
    return (
            "Please produce an *extractive* summary of the article below by choosing "
            "2 or 3 key sentences from the original text:\n\n"
            f"{article}\n\n"
            "Return only sentences from the original text that best capture the main ideas. "
            "Only write the summary and nothing else: "
        )

def ngram_cnndm(model, skip_up_to, n=5, max_len=100, samples=100):
    """
    Compute n-grams for the CNN/DailyMail dataset.
    
    Args:
        model: The model to use for generating phrases.
        skip_up_to: The number of layers to skip in the model.
        n: The size of the n-grams.
        k: The number of top phrases to consider.
        samples: The number of samples to process.
    
    Returns:
        A dictionary with n-grams and their counts.
    """
    data = cnndnn_loader()
    return_scores = []

    for id, sample in enumerate(tqdm(data)):
        if id >= samples:
            break

        # Generate the prompt for extractive summarization
        prompt = extractive_prompt(sample['article'])
        print("Prompt:", prompt)
        prompt_tokens = model.to_tokens(prompt, prepend_bos=True)

        # Get the model's output for the prompt
        og_generated = model.generate(prompt,
                                      max_new_tokens=max_len,
                                      do_sample=False,
                                      temperature=0.0,
                                      prepend_bos=True)

        # get the generated tokens
        # og_generated_tokens = og_generated[len(prompt):]

        # run once to cache the activations
        og_logits, og_cache = model.run_with_cache(prompt_tokens)

        pt_generated = prompt_tokens.clone()
        log_dict = {}
        pos_matched = []
        pos_current = []
        pt_count = 0

        # start generate and patch at the same time
        for _ in range(max_len):
            matched_pos, _ = detect_ngram_copy(pt_generated, n)

            # if no n-gram copy detected, continue
            if not matched_pos:
                pt_generated = model.generate(pt_generated,
                                                max_new_tokens=1,
                                                do_sample=False,
                                                temperature=0.0,
                                                prepend_bos=False,
                                                verbose = False)
                continue

            # if n-gram copy detected, store the positions
            pos_matched.append(matched_pos)
            pos_current.append(len(pt_generated[0]) - 1)
            
            # define the patching hook
            def residual_stream_patching_hook(
                resid_pre: Float[torch.Tensor, "batch pos d_model"],
                hook: HookPoint,
                pos_matched: list,
                pos_current: list
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                clean_resid_pre = og_cache[hook.name]
                resid_pre[:, pos_current, :] = clean_resid_pre[:, pos_matched, :]
                return resid_pre

            # create a partial function for the hook
            patching_hook = partial(residual_stream_patching_hook,
                                    pos_matched=pos_matched,
                                    pos_current=pos_current)

            # init the patched prompt tokens
            pt_logits = model.run_with_hooks(pt_generated,
                                                fwd_hooks=[
                    (utils.get_act_name("resid_pre", skip_up_to), patching_hook)
                ])

            # extract the logits with highest probabilities
            last_logit = pt_logits[:, -1, :]
            probs = torch.softmax(last_logit, dim=-1)
            top_vals, top_indices = probs.topk(1, dim=-1)
            pt_next_token = torch.tensor([top_indices[0][0]]).unsqueeze(0).to(pt_generated.device)  # ensure it's a tensor with batch dimension
            pt_generated = torch.cat([pt_generated, pt_next_token], dim=1)

            pt_count += 1

        # convert the generated tokens to string
        candidate_pt = model.to_string(pt_generated)[0][len('<|begin_of_text|>'):]
        reference_og = og_generated

        # compute the scores
        bleu_score = compute_bleu(candidate_pt, reference_og)
        rouge_l_score = compute_rouge_l(candidate_pt, reference_og)

        # log the scores
        log_dict['bleu'] = bleu_score
        log_dict['rouge_l'] = rouge_l_score
        log_dict['pt_count'] = pt_count
        log_dict['og_generated'] = og_generated
        log_dict['candidate_pt'] = candidate_pt

        return_scores.append(log_dict)
    
    return return_scores
