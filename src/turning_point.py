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

class TurningPoint:
    """
    Class for running turning point analysis on transformer models.
    """
    
    def __init__(self, model, skip_up_to, n=5, k=100, total_patched_words=20):
        """
        Initialize TurningPoint analyzer.
        
        Args:
            model: TransformerLens model.
            skip_up_to (int): Layer to apply the patch-copying intervention.
            n (int): N-gram window size for copy detection.
            k (int): Top-k for logits evaluation.
            total_patched_words (int): Maximum number of words to patch.
        """
        self.model = model
        self.skip_up_to = skip_up_to
        self.n = n
        self.k = k
        self.total_patched_words = total_patched_words
    
    def _get_model_prediction(self, prompt_tokens):
        """Get model prediction and cache for a given prompt."""
        og_logits, og_cache = self.model.run_with_cache(prompt_tokens)
        og_topk_indices = get_top_k(og_logits, self.k)
        og_next_token = torch.tensor([og_topk_indices[0]]).unsqueeze(0).to(og_logits.device)
        decoded_og_next_token = self.model.to_string(og_next_token)[0]
        return og_logits, og_cache, og_topk_indices, og_next_token, decoded_og_next_token
    
    def _compute_patch_positions(self, prompt_tokens, num_word2patch):
        """Compute positions for patching based on n-gram detection."""
        pos_matched = []
        pos_current = []
        
        for id in range(num_word2patch):
            assert id < len(prompt_tokens[0])
            match_idx, _ = detect_ngram_copy(prompt_tokens[:, :len(prompt_tokens[0])-id], n=self.n)
            pos_matched.append(match_idx)
            pos_current.append(len(prompt_tokens[0])-id-1)
        
        return pos_matched, pos_current
    
    def _residual_stream_patching_hook(self, resid_pre, hook, pos_matched, pos_current, og_cache):
        """Hook function for patching residual stream activations."""
        clean_resid_pre = og_cache[hook.name]
        resid_pre[:, pos_current, :] = clean_resid_pre[:, pos_matched, :]
        return resid_pre
    
    def _apply_patching(self, prompt_tokens, pos_matched, pos_current, og_cache):
        """Apply patching and return patched logits and predictions."""
        temp_hook_fn = partial(
            self._residual_stream_patching_hook, 
            pos_matched=pos_matched, 
            pos_current=pos_current,
            og_cache=og_cache
        )
        
        patched_logits = self.model.run_with_hooks(prompt_tokens, fwd_hooks=[
            (utils.get_act_name("resid_pre", self.skip_up_to), temp_hook_fn)
        ])
        
        pt_topk_indices = get_top_k(patched_logits, self.k)
        pt_next_token = torch.tensor([pt_topk_indices[0]]).unsqueeze(0).to(patched_logits.device)
        
        return pt_topk_indices, pt_next_token
    
    def _compute_scores(self, og_next_token, pt_next_token, og_topk_indices, pt_topk_indices, total_matches):
        """Compute accuracy and Jaccard scores."""
        if torch.equal(og_next_token, pt_next_token):
            total_matches.append(1)
        else:
            total_matches.append(0)
        
        dict_pred_info = defaultdict(dict)
        dict_pred_info[0]['original'] = og_topk_indices
        dict_pred_info[0]['copy'] = pt_topk_indices
        
        jcc, acc = get_acc([dict_pred_info])
        acc3 = sum(total_matches) / len(total_matches)
        
        return jcc, acc, acc3
    
    def ngram(self, edited_phrases, schema):
        """
        Run turning point schema for a given model and edited phrases.

        Args:
            edited_phrases (list): Dataset, each item produces a prompt via `schema`.
            schema (callable): Function that returns (corrupted_text, pre_isare, correct_tobe), (source, target).

        Returns:
            Tuple[List[Dict], List[Dict]]: Return scores and solvable cases.
        """
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
            prompt_tokens = self.model.to_tokens(prompt, prepend_bos=False)

            # Get model prediction
            og_logits, og_cache, og_topk_indices, og_next_token, decoded_og_next_token = self._get_model_prediction(prompt_tokens)

            # Only proceed if model produces the correct target
            if (target in decoded_og_next_token and correct_tobe == target and target != '') or \
               (source in decoded_og_next_token and correct_tobe == source):
                total_solvable_og += 1
            else:
                continue

            patching_succeed_flag = True
            score_list_dict = {'acc2': [], 'acc3': [], 'jcc': []}

            for num_word2patch in range(1, self.total_patched_words+1):
                if not patching_succeed_flag:
                    break

                total_matches = []
                pos_matched, pos_current = self._compute_patch_positions(prompt_tokens, num_word2patch)

                # If no valid positions found, skip patching
                if None in pos_matched or len(pos_matched) == 0:
                    total_solvable_og -= 1
                    patching_succeed_flag = False
                    break

                # Apply patching
                pt_topk_indices, pt_next_token = self._apply_patching(prompt_tokens, pos_matched, pos_current, og_cache)

                # Compute scores
                jcc, acc, acc3 = self._compute_scores(og_next_token, pt_next_token, og_topk_indices, pt_topk_indices, total_matches)
                
                score_list_dict["acc2"].append(acc)
                score_list_dict["jcc"].append(jcc)
                score_list_dict["acc3"].append(acc3)

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

    def ngram_char_edits(self, edited_phrases, schema):
        """
        Run turning point schema for a given model and edited phrases with character-level edits.

        Args:
            edited_phrases (list): List of data items for edit evaluation.
            schema (callable): Function returning {edit_type: (corrupted_sentence, decoded_up_to, ground_truth_next)}.

        Returns:
            Tuple[List[Dict], List[Dict]]: Return scores and solvable cases.
        """
        return_scores = []
        solvable_cases = []
        total_solvable_dict = {'swap': 0, 'drop': 0, 'add': 0}

        for edited in edited_phrases:
            # Stop if all edit types are solved
            if (total_solvable_dict['swap'] == 33 and 
                total_solvable_dict['drop'] == 33 and 
                total_solvable_dict['add'] == 34):
                break

            # Get outputs from the schema
            return_outputs = schema(text=edited, model=self.model)
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
                prompt_tokens = self.model.to_tokens(prompt, prepend_bos=False)

                # Get model prediction
                og_logits, og_cache, og_topk_indices, og_next_token, decoded_og_next_token = self._get_model_prediction(prompt_tokens)

                # Check if the model can solve the task
                if ground_truth_next in decoded_og_next_token:
                    total_solvable_dict[method] += 1
                else:
                    continue

                patching_succeed_flag = True
                score_list_dict = {'acc2': [], 'acc3': [], 'jcc': []}

                # Start patching the model
                for num_word2patch in range(1, self.total_patched_words+1):
                    if not patching_succeed_flag:
                        break

                    total_matches = []
                    pos_matched, pos_current = self._compute_patch_positions(prompt_tokens, num_word2patch)

                    # If no valid positions found, skip patching
                    if None in pos_matched:
                        total_solvable_dict[method] -= 1
                        patching_succeed_flag = False
                        break

                    # Apply patching
                    pt_topk_indices, pt_next_token = self._apply_patching(prompt_tokens, pos_matched, pos_current, og_cache)

                    # Compute scores
                    jcc, acc, acc3 = self._compute_scores(og_next_token, pt_next_token, og_topk_indices, pt_topk_indices, total_matches)
                    
                    score_list_dict["acc2"].append(acc)
                    score_list_dict["jcc"].append(jcc)
                    score_list_dict["acc3"].append(acc3)

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

# Legacy function wrappers for backward compatibility
def ngram(model, skip_up_to, edited_phrases, schema, n=5, k=100):
    """Legacy wrapper for the ngram function."""
    tp = TurningPoint(model, skip_up_to, n, k)
    return tp.ngram(edited_phrases, schema)

def ngram_char_edits(model, skip_up_to, edited_phrases, schema, n=5, k=100):
    """Legacy wrapper for the ngram_char_edits function."""
    tp = TurningPoint(model, skip_up_to, n, k)
    return tp.ngram_char_edits(edited_phrases, schema)
