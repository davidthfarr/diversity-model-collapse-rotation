"""Runs a repo's training loop offline with a tiny model and synthetic blocks, and records its data flow.

Usage: python tests/harness.py REPO_PATH OUTPUT_JSON [train_gen_loop arguments]
"""
import hashlib
import json
import logging
import os
import pickle
import sys
import warnings

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
repo, output = sys.argv[1:3]
sys.argv = ['train_gen_loop'] + sys.argv[3:]
sys.path.insert(0, os.path.join(repo, 'evolution'))

import lightning as L
import torch
from datasets import Dataset, DatasetDict
from transformers import GPT2Config, GPT2LMHeadModel

import helpers
import train_gen_loop

BLOCK, VOCAB = 8, 50


def split(n, seed):
    rows = torch.randint(1, VOCAB, (n, BLOCK), generator=torch.Generator().manual_seed(seed)).tolist()
    return Dataset.from_dict({'input_ids': rows, 'attention_mask': [[1] * BLOCK] * n, 'labels': rows})


def tiny_model(*_, **__):
    with torch.random.fork_rng():
        torch.manual_seed(123)
        return GPT2LMHeadModel(GPT2Config(vocab_size=VOCAB, n_positions=32, n_embd=16, n_layer=1, n_head=2))


data = DatasetDict({'train': split(160, 1), 'validation': split(16, 2), 'test': split(16, 3)})
helpers.prepare_data_wikitext2 = lambda *a, **k: data
helpers.preprocess_datasets = lambda *a, **k: data
train_gen_loop.AutoModelForCausalLM.from_pretrained = tiny_model
train_gen_loop.load_tokenizer = lambda *_: type('Tokenizer', (), {'eos_token_id': 0})()

record = {'train': [], 'prompts': [], 'train_counts': []}
original_fit, original_generate = L.Trainer.fit, GPT2LMHeadModel.generate


def fit(self, model, train_dataloaders=None, **kwargs):
    rows = [tuple(train_dataloaders.dataset[i]['input_ids'].tolist()) for i in range(len(train_dataloaders.dataset))]
    record['train'].append(rows)
    record['train_counts'].append(len(rows))
    record['prompts'].append([])
    return original_fit(self, model, train_dataloaders=train_dataloaders, **kwargs)


def generate(self, input_ids, **kwargs):
    if not record['prompts']:  # a run without training at t=0 still generates
        record['prompts'].append([])
    record['prompts'][-1] += [tuple(row) for row in input_ids.tolist()]
    return original_generate(self, input_ids, **kwargs)


L.Trainer.fit = fit
GPT2LMHeadModel.generate = generate
args = helpers.load_args_defaults()
train_gen_loop.loop_wrapper(args)

run_dir = os.path.join(args.main_dir, args.run_name)
generated, metrics, digest = {}, {}, hashlib.sha256()
for tag in sorted(os.listdir(run_dir)):
    tag_dir = os.path.join(run_dir, tag)
    for k in range(args.number_loops):
        version_dir = os.path.join(tag_dir, f'version_{k}')
        path = os.path.join(version_dir, f'dataset_version_{k}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                rows = [tuple(r['input_ids'].tolist()) for r in pickle.load(f)]
            generated[f'{tag}/{k}'] = rows
            digest.update(repr(rows).encode())
        path = os.path.join(version_dir, 'checkpoint_metrics.json')
        if os.path.exists(path):
            with open(path) as f:
                metrics[f'{tag}/{k}'] = json.load(f)
with open(output, 'w') as f:
    json.dump({**record, 'generated': generated, 'metrics': metrics, 'generated_sha256': digest.hexdigest()}, f)
