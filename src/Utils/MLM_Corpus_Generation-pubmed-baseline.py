#!/usr/bin/env python
# coding: utf-8

'''
https://www.ncbi.nlm.nih.gov/pmc/tools/amdataset/ - for why we can't use PubMed full papers (Not all articles in PMC are available for text mining and other reuse.).
we could have used - scientific_papers dataset (https://huggingface.co/datasets/scientific_papers) - however, issues as follows:
- how do we know which article corresponds to which entity - would need to run an exhaustive search for all entities on all articles. 
- The PubMed side contains articles from OA which includes The PMC Open Access Subset (or PMC OA Subset) contains millions of full-text open access article files 
made available under a Creative Commons or similar license terms or with publisher permission. This dataset includes retractions, corrections, and expressions of concern. Thus we wanted to avoid them
& focus only on accepted articles.
- this API retrieves relevant articles just based on entity names and for accepted publications.
'''

from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import pickle5 as pickle
from pymed import PubMed
import pandas as pd
import argparse
import torch


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
    pubmed = PubMed(tool="PubMedSearcher", email="placeholder@gmail.com")
    with open(args.entity_file, 'rb') as f:
        top_N_ents = pickle.load(f)

    context_dict = {}
    for ent in tqdm(top_N_ents):
        try:
            article_List = []
            full_text = []
            results = pubmed.query(ent, max_results=100)

            for article in results:
                article_Dict = article.toDict()
                article_List.append(article_Dict)

            for article in article_List:
                full_text.append(('' if article['abstract'] == None else article['abstract']) + ' ' + \
                                 ('' if article['methods'] == None else article['methods']) + ' ' + \
                                 ('' if article['results'] == None else article['results']) + ' ' + \
                                 ('' if article['conclusions'] == None else article['conclusions']))

            context_dict[ent] = full_text
        except:
            continue
        break

    print(f'Total number of entities remaining: {len(context_dict)}')
    print('Saving corpus...')
    pd.DataFrame([(key, var) for (key, L) in context_dict.items() for var in L], columns=['ent', 'text']).to_parquet(
        f'{args.corpus_file}.parquet', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity_file', default="../../data/COVID-QA/top_N_ents_spacy-COVID_QA.pkl", type=str)
    parser.add_argument('--corpus_file', default='pubmed_corpus', type=str)
    args = parser.parse_args()
    main()
