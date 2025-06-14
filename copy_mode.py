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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

torch.set_grad_enabled(False)
misc.seed_everything(555)

# load model and tokenizer
model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B", cache_dir='./hub/', device_map='auto')
# model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-3B", cache_dir='./hub/', device_map='auto')
num_layers = model.cfg.n_layers
_, _, edited_phrases = data_loader.wiki_loader(num_samples=200)

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

task_name = 'copy_mode'
info_lst = defaultdict(list)
for schema_name, schema in schemas.items():
    print(schema_name)
    for skip in tqdm(range(num_layers-1)):
        skip += 1
        if schema_name == 'char_edit':
            outputs = n_grams_cp.ngram_char_edits_cp(model= model,
                                            skip_up_to= skip,
                                            edited_phrases= edited_phrases,
                                            schema= schema)
        else:
            outputs = n_grams_cp.ngram_cp(model= model,
                            skip_up_to= skip,
                            edited_phrases= edited_phrases,
                            schema= schema)
        info_lst[schema_name].append(outputs)

    misc.save_dict_to_json(info_lst[schema_name], f"output/re_{task_name}_{schema_name}.json")