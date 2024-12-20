from datasets import load_dataset
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="Saptarshi7/covid_qa_cleaned_CS", type=str)
args = parser.parse_args()

if args.dataset == 'duorc':
  dataset = load_dataset('duorc', 'SelfRC')
  contexts = list(set(dataset['train']['plot']))
else:
  dataset = load_dataset(args.dataset, use_auth_token=True)
  contexts = list(set(dataset['train']['context']))

pd.DataFrame(zip(list(range(len(contexts))), contexts), columns=['ent', 'text']).to_csv(f"{args.dataset.replace('/','-')}_for_PPL_eval.csv", index=False)