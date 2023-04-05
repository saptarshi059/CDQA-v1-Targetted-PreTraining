# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig, logging, default_data_collator, \
    get_scheduler, set_seed
from datasets import load_dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
from evaluate import load
import EQA_Processing
import transformers
import numpy as np
import collections
import argparse
import random
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


parser = argparse.ArgumentParser()

parser.add_argument('--squad_version2', default=False, type=str2bool)
parser.add_argument('--model_checkpoint', default="roberta-base", type=str)
parser.add_argument('--trained_model_name', default="roberta-base", type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_length', default=384, type=int)
parser.add_argument('--stride', default=128, type=int)
parser.add_argument('--learning_rate', default=2e-5, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--n_best', default=20, type=int)
parser.add_argument('--max_answer_length', default=30, type=int)
parser.add_argument('--trial_mode', default=False, type=str2bool)
parser.add_argument('--random_state', default=42, type=int)
parser.add_argument('--freeze_PT_layers', default=False, type=str2bool)
parser.add_argument('--dataset_location', default='remote', type=str)
parser.add_argument('--local_dataset_name', default='Saptarshi7_covid_qa_cleaned_CS', type=str)

args = parser.parse_args()

logging.set_verbosity(50)

# Initializing all random number methods.
g = torch.Generator()
g.manual_seed(args.random_state)
torch.manual_seed(args.random_state)
random.seed(args.random_state)
set_seed(args.random_state)

squad_v2 = args.squad_version2
model_checkpoint = args.model_checkpoint
batch_size = args.batch_size

accelerator = Accelerator()
device = accelerator.device

if model_checkpoint == 'studio-ousia/luke-base':
    tokenizer = AutoTokenizer.from_pretrained(
        'roberta-base')  # since luke doesn't have a fast implementation & it has the same vocab as roberta
else:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

data_collator = default_data_collator

max_length = args.max_length  # The maximum length of a feature (question and context)
doc_stride = args.stride  # The authorized overlap between two part of the context when splitting it is needed.
max_answer_length = args.max_answer_length
n_best = args.n_best
pad_on_right = tokenizer.padding_side == "right"

if args.dataset_location == 'remote':
    if args.trial_mode:
        print('Running Code in Trial Mode to see if everything works properly...')
        raw_datasets = load_dataset("squad_v2" if squad_v2 else "squad", split=['train[:160]', 'validation[:10]'])
        train_dataset = raw_datasets[0].map(EQA_Processing.prepare_train_features(tokenizer, pad_on_right,
                                                                                  max_length, doc_stride),
                                            batched=True,
                                            remove_columns=raw_datasets[0].column_names)
        validation_dataset = raw_datasets[1].map(EQA_Processing.prepare_validation_features(tokenizer, pad_on_right,
                                                                                            max_length, doc_stride),
                                                 batched=True,
                                                 remove_columns=raw_datasets[1].column_names)
    else:
        raw_datasets = load_dataset("squad_v2" if squad_v2 else "squad")
        train_dataset = raw_datasets['train'].map(EQA_Processing.prepare_train_features(tokenizer, pad_on_right,
                                                                                        max_length, doc_stride),
                                                  batched=True,
                                                  remove_columns=raw_datasets['train'].column_names)
        validation_dataset = raw_datasets['validation'].map(EQA_Processing.prepare_validation_features(tokenizer,
                                                                                                       pad_on_right,
                                                                                                       max_length,
                                                                                                       doc_stride),
                                                            batched=True,
                                                            remove_columns=raw_datasets['validation'].column_names)

else:
    raw_dataset_train = DatasetDict({'train': load_from_disk(f'{args.local_dataset_name}_train_subset')})
    raw_dataset_validation = DatasetDict({'validation': load_from_disk(f'{args.local_dataset_name}_test_subset')})

    train_dataset = raw_dataset_train['train'].map(EQA_Processing.prepare_train_features, batched=True,
                                                   remove_columns=raw_dataset_train['train'].column_names)
    validation_dataset = raw_dataset_validation['validation'].map(EQA_Processing.prepare_validation_features,
                                                                  batched=True,
                                                                  remove_columns=
                                                                  raw_dataset_train['validation'].column_names)

metric = load("squad")

train_dataset.set_format("torch")
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size,
                              worker_init_fn=seed_worker, generator=g)

validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")
eval_dataloader = DataLoader(validation_set, collate_fn=data_collator, batch_size=batch_size,
                             worker_init_fn=seed_worker, generator=g)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
output_dir = args.trained_model_name

if args.freeze_PT_layers:
    print('Freezing base layers and only training span head...')
    base_module_name = list(model.named_children())[0][0]
    for param in getattr(model, base_module_name).parameters():
        param.requires_grad = False

optimizer = AdamW(model.parameters(), lr=args.learning_rate)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader,
                                                                          eval_dataloader)

num_train_epochs = args.epochs
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(validation_dataset)]
    end_logits = end_logits[: len(validation_dataset)]

    if args.trial_mode:
        metrics = EQA_Processing.compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets[1],
                                                 n_best, max_answer_length, metric)
    else:
        metrics = EQA_Processing.compute_metrics(start_logits, end_logits, validation_dataset,
                                                 raw_datasets['validation'], n_best, max_answer_length, metric)

    print(f"epoch {epoch}:", metrics)

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        # repo.push_to_hub(commit_message=f"Training in progress epoch {epoch}", blocking=False)
