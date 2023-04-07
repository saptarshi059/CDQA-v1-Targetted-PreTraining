# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, logging, default_data_collator, \
    get_scheduler, set_seed
from datasets import load_dataset, load_metric, DatasetDict
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
import EQA_Processing
import numpy as np
import argparse
import random
import torch

logging.set_verbosity(50)


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

parser.add_argument('--model_checkpoint', default="csarron/roberta-base-squad-v1", type=str)
parser.add_argument('--trained_model_name', default="roberta-base-squad-covidqa",
                    type=str)  # So that we can KNOW for sure which folder is what.
parser.add_argument('--batch_size', default=40, type=int)
parser.add_argument('--max_length', default=384, type=int)
parser.add_argument('--stride', default=128, type=int)
parser.add_argument('--learning_rate', default=2e-5, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--n_best', default=20, type=int)
parser.add_argument('--max_answer_length', default=1000, type=int)
parser.add_argument('--random_state', default=42, type=int)
parser.add_argument('--freeze_PT_layers', default=False, type=str2bool)

args = parser.parse_args()

g = torch.Generator()
g.manual_seed(args.random_state)
torch.manual_seed(args.random_state)
random.seed(args.random_state)
set_seed(args.random_state)

model_checkpoint = args.model_checkpoint
batch_size = args.batch_size
data_collator = default_data_collator
max_length = args.max_length  # The maximum length of a feature (question and context)
doc_stride = args.stride  # The authorized overlap between two part of the context when splitting it is needed.
max_answer_length = args.max_answer_length
n_best = args.n_best
metric = load_metric("squad")  # Since the dataset is in the same format, we can use the metrics code from squad itself

if 'luke' in model_checkpoint:
    tokenizer = AutoTokenizer.from_pretrained(
        'roberta-base')  # since luke doesn't have a fast implementation & it has the same vocab as roberta
else:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

pad_on_right = tokenizer.padding_side == "right"

raw_datasets = load_dataset("Saptarshi7/covid_qa_cleaned_CS", use_auth_token=True)

kf = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
f1_1_folds = []  # F1@1 score for each fold
em_1_folds = []  # EM@1 score for each fold

for fold_number, (train_idx, val_idx) in enumerate(kf.split(raw_datasets['train'])):
    print(f'>>>Running FOLD {fold_number + 1}<<<')

    fold_dataset = DatasetDict({"train": raw_datasets["train"].select(train_idx),
                                "validation": raw_datasets["train"].select(val_idx)})

    train_dataset = fold_dataset['train'].map(EQA_Processing.prepare_train_features, batched=True,
                                              fn_kwargs={'tokenizer': tokenizer, 'pad_on_right': pad_on_right,
                                                         'max_length': max_length, 'doc_stride': doc_stride},
                                              remove_columns=fold_dataset['train'].column_names)
    validation_dataset = fold_dataset['validation'].map(EQA_Processing.prepare_validation_features, batched=True,
                                                        fn_kwargs={'tokenizer': tokenizer, 'pad_on_right': pad_on_right,
                                                                   'max_length': max_length, 'doc_stride': doc_stride},
                                                        remove_columns=fold_dataset['validation'].column_names)

    train_dataset.set_format("torch")
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator,
                                  batch_size=batch_size, worker_init_fn=seed_worker, generator=g)
    eval_dataloader = DataLoader(validation_set, collate_fn=default_data_collator, batch_size=batch_size,
                                 worker_init_fn=seed_worker, generator=g)

    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    output_dir = f'{args.trained_model_name}-finetuned-covidqa-fold-{fold_number}'

    if args.freeze_PT_layers:
        print('Freezing base layers and only training span head...')
        base_module_name = list(model.named_children())[0][0]
        for param in getattr(model, base_module_name).parameters():
            param.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    accelerator = Accelerator()
    device = accelerator.device

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader,
                                                                              eval_dataloader)

    num_train_epochs = args.epochs
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

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

        metrics = EQA_Processing.compute_metrics(start_logits, end_logits, validation_dataset,
                                                 fold_dataset['validation'], n_best, max_answer_length, metric)

        print(f"FOLD {fold_number + 1} | EPOCH {epoch + 1}: {metrics}")

        f1_1_folds.append(metrics['f1'])
        em_1_folds.append(metrics['exact_match'])

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

# Printing Avg. EM@1 across all folds trained till a given epoch
for e in range(num_train_epochs):
    e_em = []
    for v in range(e, len(em_1_folds), num_train_epochs):
        e_em.append(em_1_folds[v])
    print(f'Average EM@1 for epoch {e} across all folds: {np.round(np.mean(e_em), 2)}')

# Printing Avg. F1@1 across all folds trained till a given epoch
for e in range(num_train_epochs):
    e_f1 = []
    for v in range(e, len(f1_1_folds), num_train_epochs):
        e_f1.append(f1_1_folds[v])
    print(f'Average F1@1 for epoch {e} across all folds: {np.round(np.mean(e_f1), 2)}')

# Printing Avg. Folds EM
counter = 0
total = 0.0
all_avg_em = []
fold_number = 1
for score in em_1_folds:
    total += score
    if counter == num_train_epochs - 1:
        print(f'Avg. EM for Fold {fold_number}: {np.round(total / num_train_epochs, 2)}')
        fold_number += 1
        all_avg_em.append(total / num_train_epochs)
        total = 0.0
        counter = 0
    else:
        counter += 1

# Printing Avg. Folds F1
counter = 0
total = 0.0
all_avg_f1 = []
fold_number = 1
for score in f1_1_folds:
    total += score
    if counter == num_train_epochs - 1:
        print(f'Avg. F1 for Fold {fold_number}: {np.round(total / num_train_epochs, 2)}')
        fold_number += 1
        all_avg_f1.append(total / num_train_epochs)
        total = 0.0
        counter = 0
    else:
        counter += 1

print(f'Avg. EM across all folds: {np.round(np.mean(all_avg_em), 2)}')
print(f'Avg. F1 across all folds: {np.round(np.mean(all_avg_f1), 2)}')
