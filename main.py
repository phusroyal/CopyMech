import argparse
from utils import data_loader, misc
from src import cont_gen, copy_mode, cnn_dm, turning_point
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="CopyMech: A Copy Mechanism for Language Models")
    
    # model name
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B',
                        help='Name of the pre-trained model to use')
    # model cache directory
    parser.add_argument('--model_cache', type=str, default='/home/anwoy/phuhoang/models/',
                        help='Directory to cache the pre-trained model')
    # number of search sentences
    parser.add_argument('--num_search_sentences', type=int, default=4_000_000,
                        help='Number of search sentences to retrieve')
    # task name
    parser.add_argument('--task_name', type=str, default='copy_mode',
                        help='Name of the task to perform')
    # solvable limit
    parser.add_argument('--solvable_limit', type=int, default=100,
                        help='Limit for solvable examples')
    # seed
    parser.add_argument('--seed', type=int, default=555,
                        help='Random seed for reproducibility')
    # device 
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on (e.g., cuda:0, cpu)')

    return parser.parse_args()

def grad_seed(args: argparse.Namespace):
    """
    Set the random seed for reproducibility and disable gradient calculations.
    
    Args:
        args: Parsed command line arguments.
    
    Returns:
        None
    """
    misc.seed_everything(args.seed)
    misc.set_grad_enabled(False)

def synth_data(args: argparse.Namespace):
    """
    Function to synthesize data based on the provided arguments.
    
    Args:
        args: Parsed command line arguments.
    
    Returns:
        None
    """
    # Here you would implement the logic to synthesize data based on the args
    print(f"Synthesizing data with model: {args.model_name}, cache: {args.model_cache}, "
          f"task: {args.task_name}, solvable limit: {args.solvable_limit}, seed: {args.seed}, "
          f"device: {args.device}")

    schemas = {
        'swap_is_are': data_loader.Scheme(source= 'is',
                                             target= 'are').swap_words,
        'swap_was_were': data_loader.Scheme(source= 'was',
                                             target= 'were').swap_words,
        'swap_a_the': data_loader.Scheme(source= 'a',
                                             target= 'the').swap_words,
        'drop_a': data_loader.Scheme(source= 'a',
                                             target= '').drop_words,
        'char_edit': data_loader.Scheme().char_edit,
    }

    grad_seed(args)
    task_name = args.task_name
    solvable_limit = args.solvable_limit
    info_lst = defaultdict(list)

    if task_name == 'copy_mode':
        task = copy_mode

    

def main():
    args = parse_args()