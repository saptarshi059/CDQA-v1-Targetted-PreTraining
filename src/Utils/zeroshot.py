import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline, set_seed


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


class QADataset(Dataset):
    def __init__(self, dataset_location):
        if dataset_location == 'remote':
            self.dataset = load_dataset('squad_v2')
        else:
            self.dataset = DatasetDict({"validation": load_dataset('json', data_files=os.path.abspath(
                "/home/saptarshi.sengupta/CDQA-v1-whole-entity-approach/data/RadQA/radqa-a-question-answering-dataset"
                "-to-improve-comprehension-of-radiology-reports-1.0.0/test.jsonl"))['train']})

        self.dataset = self.dataset.with_format("torch")

        self.samples = self.dataset['validation']
        self.questions = self.samples['question']
        self.contexts = self.samples['context']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {'question': self.questions[idx], 'context': self.contexts[idx]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', default='distilbert-base-uncased', type=str)
    parser.add_argument('--dataset_location', default='remote', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_length', default=384, type=int)
    parser.add_argument('--stride', default=128, type=int)
    parser.add_argument('--max_answer_length', default=30, type=int)
    parser.add_argument('--random_state', default=42, type=int)
    parser.add_argument('--handle_impossible_answer', default=True, type=str2bool)

    args = parser.parse_args()

    # Initializing all random number methods.
    g = torch.Generator()
    g.manual_seed(args.random_state)
    torch.manual_seed(args.random_state)
    random.seed(args.random_state)
    set_seed(args.random_state)

    model = AutoModelForQuestionAnswering.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=0)

    ds = QADataset(args.dataset_location)

    data_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

    gold_answers = [x['text'] for x in ds.samples['answers']]
    questions = ds.questions

    predicted_answers = []
    for batch in tqdm(data_loader):
        with torch.no_grad():
            predicted_answers.extend([x['answer'] for x in nlp(question=batch['question'], context=batch['context'],
                                                               max_seq_len=args.max_length, doc_stride=args.stride,
                                                               max_answer_length=args.max_answer_length,
                                                               handle_impossible_answer=args.handle_impossible_answer)])

    print('Saving predictions...')
    ds_name = 'squad_v2' if args.dataset_location == 'remote' else 'radqa'

    pd.DataFrame(zip(questions, predicted_answers, gold_answers), columns=['question', 'predictions', 'gold_answers']). \
        to_pickle(f'{args.model_checkpoint.replace("/", "_")}_{ds_name}_predictions.pkl')
