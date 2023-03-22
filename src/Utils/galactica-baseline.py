from transformers import AutoTokenizer, OPTForCausalLM, set_seed, GenerationConfig
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import argparse
import torch


class MyDataset(Dataset):
    def __init__(self, dataset):
        if dataset == 'covid-qa':
            self.data = load_dataset('Saptarshi7/covid_qa_cleaned_CS')
            self.questions = self.data['train']['question']
        else:
            self.data = load_dataset('Saptarshi7/techqa-squad-style')
            self.questions = self.data['test']['question']

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return f'Question: {self.questions[idx]}\n\nAnswer:'


def main():
    dataset = MyDataset(args.dataset)

    questions = []
    gold_answers = []
    for row in dataset.data['train']:
        questions.append(row['question'])
        gold_answers.append(row['answers']['text'])

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint: str = 'facebook/galactica-1.3b'

    generator_model = OPTForCausalLM.from_pretrained(checkpoint)
    generator_model_tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')
    generator_model.to("cuda:0")

    generator_model_tokenizer.bos_token = '<s>'
    generator_model_tokenizer.pad_token = '<pad>'
    generator_model_tokenizer.eos_token = '</s>'

    my_gen_config = GenerationConfig.from_pretrained(args.generator_model, renormalize_logits=True, do_sample=True,
                                                     max_new_tokens=args.context_max_new_answer_tokens,
                                                     top_p=0.9, temperature=0.9, use_cache=True)

    pred_answers = []
    for batch in tqdm(dataloader):
        tokenized_inputs = generator_model_tokenizer(batch, return_tensors='pt', padding=True)
        tokenized_inputs.to('cuda:0')

        with torch.no_grad():
            set_seed(args.random_seed)
            output = generator_model.generate(input_ids=tokenized_inputs['input_ids'],
                                              attention_mask=tokenized_inputs['attention_mask'],
                                              generation_config=my_gen_config)

        pred_answers.extend(generator_model_tokenizer.batch_decode(output, skip_special_tokens=True))

    print('Saving Predictions...')
    pd.DataFrame(zip(questions, pred_answers, gold_answers),
                 columns=['question', 'predictions',
                          'gold_answers']).to_pickle(f'Galactica_{args.dataset}_predictions.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="covid-qa", type=str)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--max_new_answer_tokens', default=20, type=int)
    parser.add_argument('--random_seed', default=42, type=int)
    args = parser.parse_args()
    main()
