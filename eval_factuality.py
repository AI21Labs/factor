import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# load data
def extract_example(row):
    return {'full_prefix': row.full_prefix, 'completion': row.completion,
            'contradictions': [row.contradiction_0, row.contradiction_1, row.contradiction_2]}


def read_data(path, prefix_col):
    df = pd.read_csv(path)[[prefix_col, 'doc_id', 'completion', 'contradiction_0', 'contradiction_1', 'contradiction_2']]
    df.rename(columns={prefix_col: 'full_prefix'}, inplace=True)
    return df.apply(lambda row: extract_example(row), axis=1).to_list()

# load model
def load_tokenizer(model_name, max_tokens):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right', truncation_side='left',
                                              model_max_length=max_tokens)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(model_name, cache_dir=None, max_tokens=1024):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    multi_gpus = torch.cuda.device_count() > 1
    config = AutoConfig.from_pretrained(model_name)
    model_args = {}
    if cache_dir is not None and device != 'cpu':
        model_args["cache_dir"] = cache_dir
    if multi_gpus:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_args).eval()
    if not multi_gpus:
        model = model.to(device)
    tokenizer = load_tokenizer(model_name, max_tokens)
    print(model.dtype)
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer, device

# prepare examples for evaluation
def format_data(ex):
    prefix = ex['full_prefix']
    completion = ex['completion']
    contradictions = ex['contradictions']

    # make sure completion don't contain trailing spaces
    completion = completion.lstrip(' ')
    contradictions = [cont.lstrip(' ') for cont in contradictions]

    # if the prefix ends with a new line, just concatenate.
    # Else, add space to the completion, remove it from the prefix if necessary
    if prefix.endswith(' '):
        prefix = prefix[:-1]
        batch = [f"{prefix} {completion}"] + [f"{prefix} {cont}" for cont in contradictions]
        labels_batch = [f" {completion}"] + [f" {cont}" for cont in contradictions]
    else:
        batch = [f"{prefix}{completion}"] + [f"{prefix}{cont}" for cont in contradictions]
        labels_batch = [completion] + contradictions
    return batch, labels_batch


def prep_batch(ex, tokenizer, device):
    # prepare examples for tokenization
    batch, labels_batch = format_data(ex)
    # encode full text (context + completions)
    encoding = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', add_special_tokens=False).to(device)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    input_ids = encoding['input_ids']
    # extract labels from input text
    labels_encoding = tokenizer(labels_batch, padding=True, truncation=True, return_tensors='pt', add_special_tokens=False).to(device)
    input_lens = torch.sum(encoding['attention_mask'], axis=-1).to(device)
    target_lens = torch.sum(labels_encoding['attention_mask'], axis=-1).to(device)
    offsets = input_lens - target_lens
    positions = torch.arange(0, encoding['input_ids'].size(-1))[None, :].to(device)
    labels_mask = (positions >= offsets[:, None]) * encoding['attention_mask']

    labels = input_ids*labels_mask + (-100)*(1-labels_mask)

    # assert all labels match
    for input_id, label, target_len, offset, comp in zip(input_ids, labels, target_lens, offsets, labels_batch):
        assert torch.all(input_id[offset: offset + target_len].eq(label[offset:offset+target_len])), "labels don't appear in input ids"
        assert torch.all(label[:offset] == -100), "labels include redundant prefix"
        assert torch.all(label[offset + target_len:] == -100), "labels include redundant suffix"
    encoding = {k: v.to(device) for k, v in encoding.items()}
    return encoding, labels, target_lens


def get_losses(logits, labels):
    loss_fct = CrossEntropyLoss(reduction="none")
    nll = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).cpu()
    nll = nll.view(labels.size())
    return nll


def run_eval(model, tokenizer, data, device):
    all_scores = torch.empty((len(data), 4))
    for i, ex in tqdm(enumerate(data)):
        input_ids, target, target_lens = prep_batch(ex, tokenizer, device=device)
        with torch.no_grad():
            out = model(**input_ids)
            nll = get_losses(out.logits[..., :-1, :], target[:, 1:])

        # get scores for the full the sequence
        scores = torch.sum(nll, axis=-1)
        scores = scores / target_lens.to('cpu')
        all_scores[i] = scores
        if i % 100 == 0:
            acc = np.sum(np.argmin(np.array(all_scores[:(i+1), :].tolist()), axis=1) == 0) / (i+1)
            print(f"processed: {i+1}/{len(data)} examples. accuracy: {acc}")
    return all_scores


def main(args):
    prefix_col = 'turncated_prefixes'
    data = read_data(args.data_file, prefix_col)
    model, tokenizer, device = load_model_and_tokenizer(args.model_name, args.cache_dir, max_tokens=args.max_tokens)
    all_scores = run_eval(model, tokenizer, data, device)
    data = pd.DataFrame(data)
    data['scores'] = list(all_scores.to('cpu').numpy())
    acc = np.sum(np.argmin(np.array(data['scores'].to_list()), axis=1) == 0) / len(data)
    print(f"acc = {acc}")
    data.to_json(get_results_path(args.output_folder, args.model_name), lines=True,
                 orient='records')
    print("Done!")


def get_results_path(output_folder, model_name):
    return os.path.join(output_folder, model_name.split('/')[-1] + '.jsonl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument('--data_file', required=True, type=str, help="csv file")
    parser.add_argument('--output_folder', required=True, type=str)

    # Model params
    parser.add_argument('--model_name', default='gpt2', type=str)
    parser.add_argument('--max_tokens', type=int, default=1024)

    parser.add_argument("--cache_dir", type=str, default="/dev/shm/cache-transformers/")
    args = parser.parse_args()
    main(args)
