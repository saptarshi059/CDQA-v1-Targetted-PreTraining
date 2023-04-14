#!/usr/bin/env python
# coding: utf-8

import pickle5 as pickle
from tqdm import tqdm
import pandas as pd
import wikipedia
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'none':
        return None
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity_file', default="../../data/COVID-QA/top_N_ents_spacy-COVID_QA.pkl", type=str)
    parser.add_argument('--corpus_file_name', default='wiki_corpus_covidqa_wo_filter', type=str)
    args = parser.parse_args()

    with open(args.entity_file, 'rb') as f:
        top_n_ents = pickle.load(f)

    context_dict = {}
    for ent in tqdm(top_n_ents):
        try:
            context_dict[ent] = wikipedia.page(str(ent), auto_suggest=False).content
        finally:
            continue

    print(f'Total number of entities remaining: {len(context_dict)}')
    print('Saving corpus...')
    pd.DataFrame(context_dict.items(), columns=['ent', 'text']).to_parquet(f'{args.corpus_file_name}.'
                                                                           f'parquet', index=False)


if __name__ == '__main__':
    main()
