import statistics
import argparse
import os

import pandas as pd
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from datasets import load_dataset, load_metric
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed, AutoConfig


def create_chunks(sample, chunk_size, stride):
    token_chunks = tokenizer(sample['context'], add_special_tokens=False,
                             truncation=True, max_length=chunk_size, stride=stride,
                             return_overflowing_tokens=True)['input_ids']
    blk = []
    for chunk in token_chunks:
        formatted_str = 'Question: ' + sample['question'] + ' Context: ' + tokenizer.decode(chunk) + ' Answer: '
        if len(sample['answers']['text']) != 0:
            blk.append((sample['question'], sample['answers']['text'][0], len(token_chunks), formatted_str))
        else:
            blk.append((sample['question'], '', len(token_chunks), formatted_str))

    return blk


class QADataset(Dataset):
    def __init__(self, dataset_location, dataset_split_for_radqa, chunk_size, stride):
        if dataset_location == 'remote':
            self.raw_dataset = load_dataset('Saptarshi7/covid_qa_cleaned_CS')
        else:
            self.raw_dataset = load_dataset('json', data_files=os.path.abspath(
                f'../../data/RadQA/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1'
                f'.0.0/{dataset_split_for_radqa}.jsonl'))

        self.blocks = []
        for entry in tqdm(self.raw_dataset['train']):
            self.blocks.extend(create_chunks(entry, chunk_size, stride))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        ques, gold, number_of_chunks, formatted_chunks = self.blocks[idx]
        return {'questions': ques, 'gold_answers': gold, 'number_of_chunks': number_of_chunks,
                'formatted_chunks': formatted_chunks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', default='facebook/galactica-1.3b', type=str)
    parser.add_argument('--dataset_location', default='remote', type=str)
    parser.add_argument('--dataset_split_for_radqa', default='None', type=str)
    parser.add_argument('--stride', default=128, type=int)
    parser.add_argument('--chunk_size', default=1536, type=int)
    parser.add_argument('--batch_size', default=40, type=int)
    args = parser.parse_args()

    checkpoint = args.model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_auth_token=True)
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, use_auth_token=True)

    # if checkpoint == 'facebook/galactica-1.3b':
    # model = AutoModelForCausalLM.from_pretrained(checkpoint)
    '''
    else:
    '''
    config = AutoConfig.from_pretrained(checkpoint)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()
    model = load_checkpoint_and_dispatch(model, os.path.abspath('/home/saptarshi.sengupta/.cache/huggingface/hub'
                                                                '/models--epfl-llm--meditron-7b/snapshots'
                                                                '/9f16d7596f37de958bd3e6812dc4584eaf86cd71'),
                                         device_map="auto", no_split_module_classes=["LlamaDecoderLayer"])

    print(f'Model:{checkpoint} loaded...')
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device_map='auto')

    dataset = QADataset(args.dataset_location, args.dataset_split_for_radqa, args.chunk_size, args.stride)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    generator.tokenizer.pad_token_id = model.config.eos_token_id

    questions = []
    gold_answers = []
    pred_answers = []
    no_chunks = []

    print('Computing Predictions...')
    for batch in tqdm(data_loader):
        questions.extend(batch['questions'])
        d = [[elem] for elem in batch['gold_answers']]
        gold_answers.extend(d)
        no_chunks.extend(batch['number_of_chunks'].tolist())

        set_seed(42)
        generations = generator(batch['formatted_chunks'], renormalize_logits=True, do_sample=True,
                                max_new_tokens=50, top_p=0.9, temperature=0.9, use_cache=True)

        for gen in generations:
            pred_answers.append(gen[0]['generated_text'].split('Answer:', 1)[1].strip())

    print('Computing Scores...')
    df = pd.DataFrame(zip(questions, no_chunks, pred_answers, gold_answers),
                      columns=['questions', 'no_chunks', 'predictions', 'gold_answers'])

    if args.dataset_location == 'remote':
        metric = load_metric('squad')

        EM = []
        F1 = []

        for row in df.itertuples():
            pred = [{"id": str(row.Index), "prediction_text": row.predictions}]
            true = [{"id": str(row.Index), "answers": {'answer_start': [1 for i in range(len(row.gold_answers))],
                                                       'text': row.gold_answers}}]

            metrics = metric.compute(predictions=pred, references=true)

            EM.append(metrics['exact_match'])
            F1.append(metrics['f1'])

        df['EM'] = EM
        df['F1'] = F1

        curr_row = 0

        max_EM = []
        max_F1 = []

        while curr_row < df.shape[0]:
            offset = df.iloc[curr_row]['no_chunks']
            sub_df = df.iloc[curr_row:curr_row + offset]
            max_values = sub_df.max(axis='rows', numeric_only=True)
            max_EM.append(max_values['EM'])
            max_F1.append(max_values['F1'])
            curr_row = curr_row + offset

        print(f'Avg. EM: {df.loc[:, "EM"].mean()} | Avg. F1: {df.loc[:, "F1"].mean()} | '
              f'Avg. best EM:{statistics.fmean(max_EM)} | '
              f'Avg. best F1: {statistics.fmean(max_F1)}')

    else:
        metric = load_metric('squad_v2')
        pred = []
        true = []
        for row in df.itertuples():
            if len(str(row.predictions)) == 0:
                pred.append({"id": str(row.Index), "prediction_text": "", 'no_answer_probability': 1.})
            else:
                pred.append({"id": str(row.Index), "prediction_text": row.predictions, 'no_answer_probability': 0.})

            if len(row.gold_answers) == 0:
                true.append({"id": str(row.Index), "answers": {'answer_start': [1], 'text': []}})
            else:
                true.append({"id": str(row.Index),
                             "answers": {'answer_start': [1 for i in range(len(row.gold_answers))],
                                         'text': row.gold_answers}})

        metrics = metric.compute(predictions=pred, references=true)

        print(f'EM: {metrics["exact"]} | F1: {metrics["f1"]} | '
              f'EM_has_ans: {metrics["HasAns_exact"]} | F1_has_ans: {metrics["HasAns_f1"]}')
