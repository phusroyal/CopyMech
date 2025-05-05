import torch, random, numpy as np, json, os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def save_dict_to_json(data, filename):
    """Saves a dictionary to a JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def load_dict_from_json(filename):
    """Loads a dictionary from a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)
    
def load_json_files(folder_path):
    """
    Loads all JSON files in the specified folder and returns their contents as a list.

    Parameters:
        folder_path (str): Path to the folder containing JSON files.

    Returns:
        list: A list where each element is the data loaded from a JSON file.
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_top_k(logits, top_k=5):
    """
    logits: (B, T, vocab_size)
    Returns a list of top-k token IDs for the last position, e.g. [id1, id2,...].
    """
    last_logits = logits[:, -1, :]       # shape (B, vocab_size)
    probs = torch.softmax(last_logits, dim=-1)
    top_vals, top_indices = probs.topk(top_k, dim=-1)
    # top_indices is shape (B, top_k). For B=1, we do top_indices[0].tolist().
    return top_indices[0].tolist()

def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return  intersection / union

def get_acc(info_lst, return_all=False):
    jcc_ult = []
    acc_ult = []

    for data in info_lst:
        acc_lst = []
        jc_lst = []
        for step in data.keys():
            copy = data[step]['copy']
            original = data[step]['original']

            jaccard_score = jaccard_similarity(copy, original)
            jc_lst.append(jaccard_score)

            acc_score = 1 if copy[0] == original[0] else 0
            acc_lst.append(acc_score)

        jcc_ult.append(jc_lst)
        acc_ult.append(acc_lst)
    
    def cal_avg(lsts):
        avg_lst = []
        for lst in lsts:
            if len(lst) == 0:
                continue
            avg_lst.append(sum(lst) / len(lst))
        return sum(avg_lst) / len(avg_lst)

    avg_jcc = cal_avg(jcc_ult)
    avg_acc = cal_avg(acc_ult)

    if return_all:
        return avg_jcc, avg_acc, jcc_ult, acc_ult

    return avg_jcc, avg_acc

def compute_bleu(candidate, reference):
    """
    Compute BLEU score for a candidate sentence against a reference sentence.
    Both candidate and reference should be provided as lists of tokens.
    """
    # NLTK expects reference to be a list of reference sentences (each a list of tokens)
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([reference], candidate, smoothing_function=smoothie)
    return bleu

def lcs_length(x, y):
    """
    Compute the length of the longest common subsequence between two sequences x and y.
    x and y are lists of tokens.
    """
    m = len(x)
    n = len(y)
    # Create a DP table of size (m+1) x (n+1)
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
    Compute ROUGE-L F-measure between a candidate and a reference sentence.
    
    ROUGE-L is based on the length of the longest common subsequence (LCS) between the candidate
    and reference. The F-measure is computed as:
    
        F = ((1 + beta^2) * P * R) / (R + beta^2 * P)
    
    where:
        P = LCS(candidate, reference) / len(candidate)
        R = LCS(candidate, reference) / len(reference)
    
    Parameters:
      candidate (list): Candidate sentence as a list of tokens.
      reference (list): Reference sentence as a list of tokens.
      beta (float): Weighting factor (default 1.0 gives equal weight to precision and recall).
    
    Returns:
      float: The ROUGE-L F1 score.
    """
    lcs = lcs_length(candidate, reference)
    precision = lcs / len(candidate) if candidate else 0.0
    recall = lcs / len(reference) if reference else 0.0
    if precision + recall == 0:
        fscore = 0.0
    else:
        fscore = ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)
    return fscore


def plot_skip_layer_metrics(skip_layers, accuracy3, jaccard_similarity, model_name, schema, leg_loc='lower right'):
    """
    Plots a grouped bar chart of Accuracy 3 and Jaccard Similarity for different skip layers.
    
    Parameters:
      skip_layers (list): List of skip layer values (e.g., [5, 10, 15, 20, 25, 30]).
      accuracy3 (list): List of Accuracy 3 values corresponding to each skip layer.
      jaccard_similarity (list): List of Jaccard similarity values corresponding to each skip layer.
    """
    # Create an array with the positions for each skip layer on the x-axis.
    x = np.arange(len(skip_layers))
    width = 0.35  # width of each bar

    # Create the plot and two sets of bars.
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_acc = ax.bar(x - width/2, accuracy3, width, label='Accuracy')
    bars_jacc = ax.bar(x + width/2, jaccard_similarity, width, label='Jaccard Similarity')

    # Label the axes and add a title
    ax.set_xlabel('Skip Layers')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'{schema}: {model_name}\'s Accuracy and Jaccard Similarity by Skip Layers')
    ax.set_xticks(x)
    ax.set_xticklabels(skip_layers)
    ax.legend(loc = leg_loc)

    # Add numerical labels above the bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(bars_acc)
    autolabel(bars_jacc)

    plt.tight_layout()
    plt.show()

def elementwise_mean(matrices):
    # Stack the matrices along a new axis, then compute the mean along that axis.
    matrices = np.array(matrices)
    stacked = np.stack(matrices)
    return np.mean(stacked, axis=0)

def get_mat(data_lst, score_types, num_items, num_layers):        
    # Compute the mean matrix for each score type
    matrices = {}

    for score in score_types:
        score_data = []
        for data in data_lst:
            matrix = np.zeros((num_layers, num_items))
            for layer_idx, layer in enumerate(data):
                for item_idx in range(num_items):
                    # For each dictionary in the current layer, extract the score at item_idx
                    values = [d[score][item_idx] for d in layer]
                    matrix[layer_idx, item_idx] = np.mean(values)
            score_data.append(matrix)            
        matrices[score] = elementwise_mean(score_data)
    return matrices

def plot_score_heatmaps(inputs, score_types=['acc2']):
    """
    Given data as a list of layers, where each layer is a list of dictionaries,
    and each dictionary has keys (e.g., "acc2", "acc3", "jcc") mapping to a list of scores
    of fixed length (num_items), this function computes, for each score type,
    a matrix of shape (num_layers, num_items) with the mean values (across the dictionaries per layer)
    and plots a heatmap with the y-axis representing layers (lower layers at the bottom)
    and the x-axis representing items.
    
    Colors are mapped from 0 (red) to 1 (green) using the RdYlGn colormap.
    """  
    title_dict= {
        'acc2': 'Accuracy',
        'jcc': 'Jaccard Distance'
    }

    num_items = 20
    num_layers = len(inputs[0])

    matrices = get_mat(data_lst= inputs,
                       score_types= score_types,
                       num_items= num_items,
                       num_layers= num_layers)

    # Create a subplot for each score type.
    fig, axs = plt.subplots(1, len(score_types), figsize=(10, 9))
    if len(score_types) == 1:
        axs = [axs]

    for ax, score in zip(axs, score_types):
        # Flip the matrix vertically so that the lowest layer (index 0) is at the bottom.
        matrix_flipped = matrices[score][::-1, :]
        
        # Create the heatmap using seaborn.
        # For x-axis, we label from 1 to num_items.
        # For y-axis, since we flipped the matrix, label from num_layers down to 1.
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