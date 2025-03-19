import torch, random, numpy as np, json
import matplotlib.pyplot as plt
import seaborn as sns

def save_dict_to_json(data, filename):
    """Saves a dictionary to a JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def load_dict_from_json(filename):
    """Loads a dictionary from a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)

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

def plot_score_heatmaps(input, schema_name, score_types=['acc2']):
    """
    Given data as a list of layers, where each layer is a list of dictionaries,
    and each dictionary has keys (e.g., "acc2", "acc3", "jcc") mapping to a list of scores
    of fixed length (num_items), this function computes, for each score type,
    a matrix of shape (num_layers, num_items) with the mean values (across the dictionaries per layer)
    and plots a heatmap with the y-axis representing layers (lower layers at the bottom)
    and the x-axis representing items.
    
    Colors are mapped from 0 (red) to 1 (green) using the RdYlGn colormap.
    """
    data = input[schema_name]
    num_layers = len(data)       # number of layers
    num_items = 20               # each score is a list of length num_items
    
    # Compute the mean matrix for each score type
    matrices = {}
    for score in score_types:
        matrix = np.zeros((num_layers, num_items))
        for layer_idx, layer in enumerate(data):
            for item_idx in range(num_items):
                # For each dictionary in the current layer, extract the score at item_idx
                values = [d[score][item_idx] for d in layer]
                matrix[layer_idx, item_idx] = np.mean(values)
        matrices[score] = matrix

    # Create a subplot for each score type.
    fig, axs = plt.subplots(1, len(score_types), figsize=(7, 9))
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
                    vmin=0, vmax=1,
                    cmap='RdYlGn',
                    annot=True,
                    # fmt=".2f",
                    xticklabels=[f"{i+1}" for i in range(num_items)],
                    yticklabels=[f"{i}" for i in range(num_layers, 0, -1)]
                   )
        ax.set_title(score)
    
    plt.tight_layout()
    plt.show()