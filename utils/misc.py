import torch, random, numpy as np
import time
import torch
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


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

def get_acc(info_lst):
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

    return avg_jcc, avg_acc


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