__author__ = 'Connor Heaton'

import re
import os
import math
import time
import argparse

import pandas as pd
import pickle5 as pickle

from tqdm.auto import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed


class PromptDataset(Dataset):
    def __init__(self, entities, prompt_string, n_context_per_entity):
        self.items = []
        for entity in entities:
            for _ in range(n_context_per_entity):
                self.items.append([entity, prompt_string.format(entity)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        entity_str, prompt = self.items[idx]

        return {'entity': entity_str, 'prompt': prompt}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_model', default="facebook/galactica-1.3b", type=str)
    parser.add_argument('--entity_file', default="../../data/COVID-QA/ents_spacy.pkl", type=str)
    parser.add_argument('--context_max_len', default=2048, type=int)
    parser.add_argument('--n_context_per_entity', default=1, type=int)
    parser.add_argument('--prompt_string', default='Title: {}', type=str)

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int, help='zero-indexed')
    parser.add_argument('--out', default='../../out/gen_v1')
    parser.add_argument('--summary_every', default=1, type=int)

    parser.add_argument('--batch_size', default=-1, type=int)

    args = parser.parse_args()

    if args.batch_size < 0:
        args.batch_size = args.n_context_per_entity

    if not os.path.exists(args.out) and args.rank == 0:
        os.makedirs(args.out)

    out_fp = os.path.join(args.out, 'rank{}_gens.parquet'.format(args.rank))

    print('[rank {}] Reading entities'.format(args.rank))
    ents_file_path = os.path.abspath(args.entity_file)
    with open(ents_file_path, 'rb') as f:
        ents_main = pickle.load(f)

    print('ents_main[:10]: {}'.format(ents_main[:10]))

    n_ents = len(ents_main)
    ents_per_rank = int(math.ceil(n_ents / args.world_size))
    rank_ents = ents_main[args.rank * ents_per_rank: (args.rank + 1) * ents_per_rank]
    print('[rank {}] n_ents: {}'.format(args.rank, n_ents))
    print('[rank {}] ents_per_rank: {}'.format(args.rank, ents_per_rank))
    print('[rank {}] len(rank_ents): {}'.format(args.rank, len(rank_ents)))

    print('[rank {}] Making pipeline...'.format(args.rank, len(rank_ents)))
    generator_model = AutoModelForCausalLM.from_pretrained(args.teacher_model)
    generator_model_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)

    generator = pipeline('text-generation', model=generator_model, tokenizer=generator_model_tokenizer,
                         device=args.rank)
    generator.tokenizer.pad_token_id = generator_model.config.eos_token_id

    print('[rank {}] Making dataset...'.format(args.rank))
    dataset = PromptDataset(entities=rank_ents, prompt_string=args.prompt_string,
                            n_context_per_entity=args.n_context_per_entity)
    print('[rank {}] len(dataset): {}'.format(args.rank, len(dataset)))
    n_iters = int(math.ceil(len(dataset) / args.batch_size))

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    write_data = []
    start_time = time.time()
    for batch_idx, batch_data in enumerate(data_loader):
        entity_strs = batch_data['entity']
        entity_prompts = batch_data['prompt']

        set_seed(42)
        generations = generator(
            entity_prompts, renormalize_logits=True, do_sample=True, max_length=args.context_max_len,
            top_p=0.9, temperature=0.9, use_cache=True, batch_size=args.batch_size
        )
        generations = [gen[0]['generated_text'] for gen in generations]
        batch_write_data = list(zip(entity_strs, entity_prompts, generations))
        write_data.extend(batch_write_data)

        if batch_idx % args.summary_every == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / (batch_idx + 1)
            avg_time_per_context = elapsed_time / (len(write_data))
            print_str = '[rank {0}] Generated contexts for batch {1}/{2} - {3:.2f}s/batch {4:.2f}s/context'.format(
                args.rank, batch_idx + 1, n_iters, avg_time_per_batch, avg_time_per_context
            )
            print(print_str)

    print('[rank {}] Saving data... out_fp={}'.format(args.rank, out_fp))
    entity_idxs, titles, generations = map(list, zip(*write_data))
    write_df = pd.DataFrame(zip(entity_idxs, titles, generations),
                            columns=['entity', 'prompt', 'context'])
    # print(write_df)
    write_df.to_parquet(out_fp)
    print('[rank {}] Done :)'.format(args.rank))
