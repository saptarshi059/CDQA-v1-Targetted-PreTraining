#!/usr/bin/env python
# coding: utf-8

from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import random
import torch
import math

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', default="csarron/roberta-base-squad-v1", type=str)
parser.add_argument('--random_state', default=42, type=int)
args = parser.parse_args()






g = torch.Generator()
g.manual_seed(args.random_state)
torch.manual_seed(args.random_state)
random.seed(args.random_state)

corpus_dataset = load_dataset("csv", data_files="data/corpus.csv")

model_checkpoint = args.model_checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

#Adding the new tokens to the vocabulary
print(f'Original number of tokens: {len(tokenizer)}')
new_tokens = corpus_dataset['train']['ent']
tokenizer.add_tokens(new_tokens)
print(f'New number of tokens: {len(tokenizer)}')

model.resize_token_embeddings(len(tokenizer)) 
# The new vector is added at the end of the embedding matrix

tokenized_datasets = corpus_dataset['train'].select([0,1,2]).map(tokenize_function, batched=True, remove_columns=['split', 'ent', 'text'])
chunk_size = tokenizer.model_max_length
lm_datasets = tokenized_datasets.map(group_texts, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

batch_size = 64
train_dataloader = DataLoader(lm_datasets, shuffle=True, batch_size=batch_size, collate_fn=data_collator, worker_init_fn=seed_worker, generator=g)

optimizer = AdamW(model.parameters(), lr=5e-5)

accelerator = Accelerator()
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

output_dir = f'trained_model/{model_checkpoint.replace('/','-')}_new_tokens'

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    print(f">>> Epoch {epoch} Complete")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

print('Training done and model saved')