#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import pickle5 as pickle
import pandas as pd
import wikipedia
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity_file', default="../../data/COVID-QA/top_N_ents_spacy-COVID_QA.pkl", type=str)
    parser.add_argument('--filtering', default=False, type=str2bool)
    parser.add_argument('--corpus_file', default='wiki_corpus_covidqa_wo_filter', type=str)
    parser.add_argument('--device', default=0, type=int)
    args = parser.parse_args()

    with open(args.entity_file, 'rb') as f:
        top_N_ents = pickle.load(f)

    search_res = {}
    for ent in tqdm(top_N_ents):
        #Skipping those entities which don't return anything
        if wikipedia.search(ent) != []:
            search_res[ent] = wikipedia.search(str(ent), results=1)[0]
        break
    print(f'Number of entities after 1st round of filtering (removing empty wiki results): {len(search_res)}')

    context_dict = {}
    filtering = args.filtering
    print(f'Filtering using semantic similarity: {filtering}')
    if filtering == True:
        filtering_threshold = 0.5
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        
        #We're consider scibert because pubmedbert assigns very high similarity for both related/unrelated terms.
        checkpoint = 'allenai/scibert_scivocab_uncased'
        
        model = AutoModel.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        cos = torch.nn.CosineSimilarity(dim=0)
        model.to(device)

    for ent, res in tqdm(search_res.items()):
        if filtering == False:
            try:
                context_dict[ent] = wikipedia.page(res, auto_suggest=False).content          
            except:
                continue
        else:
            encoded_input = tokenizer([ent, res], return_tensors='pt', padding=True)
            with torch.no_grad():
                output = model(**encoded_input)
            
            similarity = cos(output.pooler_output[0], output.pooler_output[1])

            #we're taking less than here since the similarity scores for related terms seem to be lower than unrelated ones.
            if similarity.item() < filtering_thresold:
                try:
                    context_dict[ent] = wikipedia.page(res, auto_suggest=False).content          
                except:
                    continue

    print(f'Total number of entities remaining: {len(context_dict)}')
    print('Saving corpus...')
    pd.DataFrame(context_dict.items(), columns = ['ent', 'text']).to_parquet(f'{args.corpus_file}.parquet', index=False)

if __name__ == '__main__':
    main()