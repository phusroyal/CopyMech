import torch ,os
from utils import data_loader, misc, n_grams_cp
from collections import defaultdict
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModel
try:
    from access_token import access_token
    os.environ["HF_TOKEN"] = access_token

except ImportError:
    print("No access token loaded.")

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# token = '_'
# os.environ["HUGGINGFACE_HUB_TOKEN"] = token
# os.environ["HF_TOKEN"] = token  # Alternative variable name

os.environ["https_proxy"] = "http://xen03.iitd.ac.in:3128"
os.environ["http_proxy"] = "http://xen03.iitd.ac.in:3128"


torch.set_grad_enabled(False)
misc.seed_everything(555)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model and tokenizer
model_cache = '/home/anwoy/phuhoang/models/'
model_name = "meta-llama/Llama-3.2-3B"  # or "Qwen/Qwen2.5-3B"
model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B", cache_dir=model_cache, device=device)

num_layers = model.cfg.n_layers
_, _, edited_phrases = data_loader.wiki_loader(num_samples=4_000_000)

schemas = {
    # 'swap_is_are': data_loader.Scheme(source= 'is',
    #                                      target= 'are').swap_words,
    # 'swap_was_were': data_loader.Scheme(source= 'was',
    #                                      target= 'were').swap_words,
    # 'swap_a_the': data_loader.Scheme(source= 'a',
    #                                      target= 'the').swap_words,
    # 'drop_a': data_loader.Scheme(source= 'a',
    #                                      target= '').drop_words,
    'char_edit': data_loader.Scheme().char_edit,
}

task_name = 'copy_mode'
solvable_limit = 100  # Limit for solvable examples
info_lst = defaultdict(list)
for schema_name, schema in schemas.items():
    print(schema_name)
    for skip in tqdm(range(num_layers-1)):
        if schema_name == 'char_edit':
            outputs = n_grams_cp.ngram_char_edits_cp(model= model,
                                            skip_up_to= skip+1,
                                            edited_phrases= edited_phrases,
                                            schema= schema,
                                            solvable_limit= solvable_limit)
        else:
            outputs = n_grams_cp.ngram_cp(model= model,
                            skip_up_to= skip,
                            edited_phrases= edited_phrases,
                            schema= schema,
                            solvable_limit= solvable_limit)
        info_lst[schema_name].append(outputs)

    misc.save_dict_to_json(info_lst[schema_name], f"output/{model_name}/{task_name}/{task_name}_{schema_name}.json")