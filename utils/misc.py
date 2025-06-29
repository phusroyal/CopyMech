import torch
import random
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def save_dict_to_json(data, filename):
    """Saves a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def load_dict_from_json(filename):
    """Load a dictionary from a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def load_json_files(folder_path):
    """
    Load all JSON files from a folder and return their contents as a list.
    
    Args:
        folder_path (str): Path to the folder containing JSON files.
    
    Returns:
        list: List of data loaded from each JSON file.
    """
    data_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                data_list.append(data)
    return data_list


def seed_everything(seed):
    """Set random seeds for reproducible results across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_ngram_copy(seq_ids: torch.Tensor, n=3, skip_up_to=43):
    """
    Try to find a scenario where the last n-gram in the input sequence is a copy
    of an earlier n-gram in the same sequence. Used for patch-copying detection.

    Args:
        seq_ids (torch.Tensor): Input sequence tensor of shape (1, T).
        n (int): Size of n-gram to match.
        skip_up_to (int): Layer index parameter (passed through for interface compatibility).

    Returns:
        tuple: (matched_pos, skip_up_to) or (None, None) if not found.
    """
    T = seq_ids.size(1)
    if T < n:
        return None, None

    # Check if the last token can be matched
    # with an earlier n-gram of size n-1
    last_token = seq_ids[0, -1].item()
    possible_pos = (seq_ids[0, :-1] == last_token).nonzero().view(-1)
    if possible_pos.numel() == 0:
        return None, None

    # We need to match the last n-1 tokens
    # with an earlier n-gram of size n-1
    n_minus_1 = n - 1
    context_needed = seq_ids[0, -(n_minus_1+1):-1]  # last n-1 tokens

    # Check if any of the possible positions
    # match the context needed
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


def get_top_k(logits, top_k=5):
    """
    Extract top-k token IDs from model logits for the last position.
    
    Args:
        logits (torch.Tensor): Model logits of shape (B, T, vocab_size)
        top_k (int): Number of top tokens to return
    
    Returns:
        list: Top-k token IDs for the last position
    """
    last_logits = logits[:, -1, :]       # shape (B, vocab_size)
    probs = torch.softmax(last_logits, dim=-1)
    top_vals, top_indices = probs.topk(top_k, dim=-1)
    # Assuming batch size of 1, return the top-k indices as a list
    return top_indices[0].tolist()


def jaccard_similarity(list1, list2):
    """
    Calculate Jaccard similarity coefficient between two lists.
    
    Args:
        list1, list2 (list): Input lists to compare
    
    Returns:
        float: Jaccard similarity score (intersection/union)
    """
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union if union > 0 else 0.0


def compute_bleu(candidate, reference):
    """
    Compute BLEU score between candidate and reference sentences.
    
    Args:
        candidate (list): Candidate sentence as list of tokens
        reference (list): Reference sentence as list of tokens
    
    Returns:
        float: BLEU score
    """
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([reference], candidate, smoothing_function=smoothie)
    return bleu


def lcs_length(x, y):
    """
    Compute the length of the longest common subsequence using dynamic programming.
    
    Args:
        x, y (list): Input sequences to compare
    
    Returns:
        int: Length of the longest common subsequence
    """
    m, n = len(x), len(y)
    # DP table: dp[i][j] = LCS length of x[:i] and y[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    
    return dp[m][n]


def compute_rouge_l(candidate, reference, beta=1.0):
    """
    Compute ROUGE-L F-measure based on longest common subsequence.
    
    ROUGE-L measures the longest common subsequence between candidate and reference.
    F-measure formula: F = ((1 + β²) * P * R) / (R + β² * P)
    where P = precision, R = recall
    
    Args:
        candidate (list): Candidate sentence as list of tokens
        reference (list): Reference sentence as list of tokens
        beta (float): Weight factor for F-measure (default: 1.0 for F1)
    
    Returns:
        float: ROUGE-L F-measure score
    """
    lcs = lcs_length(candidate, reference)
    precision = lcs / len(candidate) if candidate else 0.0
    recall = lcs / len(reference) if reference else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    fscore = ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)
    return fscore


def get_acc(info_lst, return_all=False):
    """
    Calculate average accuracy and Jaccard similarity from evaluation data.
    
    Args:
        info_lst (list): List of evaluation data dictionaries
        return_all (bool): Whether to return individual scores in addition to averages
    
    Returns:
        tuple: (avg_jaccard, avg_accuracy) or (avg_jaccard, avg_accuracy, all_jaccard, all_accuracy)
    """
    jcc_ult = []  # All Jaccard scores
    acc_ult = []  # All accuracy scores

    for data in info_lst:
        acc_lst = []
        jc_lst = []
        
        for step in data.keys():
            copy = data[step]['copy']
            original = data[step]['original']

            # Calculate Jaccard similarity
            jaccard_score = jaccard_similarity(copy, original)
            jc_lst.append(jaccard_score)

            # Calculate top-1 accuracy (first token match)
            acc_score = 1 if copy[0] == original[0] else 0
            acc_lst.append(acc_score)

        jcc_ult.append(jc_lst)
        acc_ult.append(acc_lst)
    
    def cal_avg(lsts):
        """Helper function to calculate average across lists of lists."""
        avg_lst = []
        for lst in lsts:
            if len(lst) == 0:
                continue
            avg_lst.append(sum(lst) / len(lst))
        return sum(avg_lst) / len(avg_lst) if avg_lst else 0.0

    avg_jcc = cal_avg(jcc_ult)
    avg_acc = cal_avg(acc_ult)

    if return_all:
        return avg_jcc, avg_acc, jcc_ult, acc_ult
    return avg_jcc, avg_acc


def plot_skip_layer_metrics(skip_layers, accuracy3, jaccard_similarity, model_name, schema, leg_loc='lower right'):
    """
    Create a grouped bar chart comparing accuracy and Jaccard similarity across skip layers.
    
    Args:
        skip_layers (list): Skip layer values for x-axis
        accuracy3 (list): Accuracy values for each skip layer
        jaccard_similarity (list): Jaccard similarity values for each skip layer
        model_name (str): Name of the model for title
        schema (str): Schema name for title
        leg_loc (str): Legend location
    """
    x = np.arange(len(skip_layers))
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_acc = ax.bar(x - width/2, accuracy3, width, label='Accuracy')
    bars_jacc = ax.bar(x + width/2, jaccard_similarity, width, label='Jaccard Similarity')

    # Formatting
    ax.set_xlabel('Skip Layers')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'{schema}: {model_name}\'s Accuracy and Jaccard Similarity by Skip Layers')
    ax.set_xticks(x)
    ax.set_xticklabels(skip_layers)
    ax.legend(loc=leg_loc)

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(bars_acc)
    autolabel(bars_jacc)

    plt.tight_layout()
    plt.show()


def elementwise_mean(matrices):
    """Calculate element-wise mean across a list of matrices."""
    matrices = np.array(matrices)
    stacked = np.stack(matrices)
    return np.mean(stacked, axis=0)


def get_mat(data_lst, score_types, num_items, num_layers):
    """
    Extract and average score matrices from evaluation data.
    
    Args:
        data_lst (list): List of evaluation data
        score_types (list): Types of scores to extract
        num_items (int): Number of items per layer
        num_layers (int): Number of layers
    
    Returns:
        dict: Matrices for each score type
    """
    matrices = {}

    for score in score_types:
        score_data = []
        
        for data in data_lst:
            matrix = np.zeros((num_layers, num_items))
            
            for layer_idx, layer in enumerate(data):
                for item_idx in range(num_items):
                    # Extract scores for current item across all dictionaries in layer
                    values = [d[score][item_idx] for d in layer]
                    matrix[layer_idx, item_idx] = np.mean(values)
            
            score_data.append(matrix)
        
        # Average across all data instances
        matrices[score] = elementwise_mean(score_data)
    
    return matrices


def plot_score_heatmaps(inputs, score_types=['acc2']):
    """
    Generate heatmaps showing score distributions across layers and items.
    
    Creates heatmaps where:
    - Y-axis: Model layers (bottom = layer 1, top = highest layer)
    - X-axis: Items (1 to num_items)
    - Colors: Score values (red = low, green = high)
    
    Args:
        inputs (list): Evaluation data structured as [data][layer][dict][score][item]
        score_types (list): Score types to plot (e.g., ['acc2', 'jcc'])
    """
    # Score type display names
    title_dict = {
        'acc2': 'Accuracy',
        'jcc': 'Jaccard Similarity'
    }

    num_items = 20
    num_layers = len(inputs[0])

    # Extract score matrices
    matrices = get_mat(data_lst=inputs,
                      score_types=score_types,
                      num_items=num_items,
                      num_layers=num_layers)

    # Create subplots
    fig, axs = plt.subplots(1, len(score_types), figsize=(10, 9))
    if len(score_types) == 1:
        axs = [axs]

    for ax, score in zip(axs, score_types):
        # Flip matrix so layer 1 appears at bottom
        matrix_flipped = matrices[score][::-1, :]
        
        # Create heatmap
        sns.heatmap(matrix_flipped,
                    ax=ax,
                    vmin=0.4, vmax=1,
                    cmap='RdYlGn',
                    annot=True,
                    # fmt=".2f",
                    xticklabels=[f"{i+1}" for i in range(num_items)],
                    yticklabels=[f"{i}" for i in range(num_layers, 0, -1)]
                   )
        ax.set_title(title_dict[score])
    
    plt.tight_layout()
    plt.show()