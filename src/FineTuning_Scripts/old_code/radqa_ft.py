# -*- coding: utf-8 -*-

import argparse
import collections
import random

import numpy as np
import torch
import os
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
from evaluate import load
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, logging, default_data_collator, \
    get_scheduler, set_seed


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


def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace

    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        if 't5' not in model_checkpoint:
            cls_index = input_ids.index(tokenizer.cls_token_id)
        else:
            cls_index = input_ids.index(tokenizer.eos_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"],
                                      "no_answer_probability": 0.0})
        else:
            predicted_answers.append({"id": example_id, "prediction_text": "", "no_answer_probability": 1.0})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


parser = argparse.ArgumentParser()

parser.add_argument('--model_checkpoint', default="distilbert-base-uncased", type=str)
parser.add_argument('--trained_model_name', default="distilbert-base-uncased-radqa", type=str)
parser.add_argument('--dataset_location', default="../../../data/RadQA/radqa-a-question-answering-dataset-to-improve"
                                                  "-comprehension-of-radiology-reports-1.0.0/", type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_length', default=384, type=int)
parser.add_argument('--stride', default=128, type=int)
parser.add_argument('--learning_rate', default=3e-5, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--n_best', default=20, type=int)
parser.add_argument('--max_answer_length', default=1000, type=int)
parser.add_argument('--trial_mode', default=False, type=str2bool)
parser.add_argument('--random_state', default=42, type=int)

args = parser.parse_args()

logging.set_verbosity(50)

# Initializing all random number methods.
g = torch.Generator()
g.manual_seed(args.random_state)
torch.manual_seed(args.random_state)
random.seed(args.random_state)
set_seed(args.random_state)

model_checkpoint = args.model_checkpoint
batch_size = args.batch_size

accelerator = Accelerator()
device = accelerator.device

if 'luke' in model_checkpoint:
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

if args.trial_mode:
    print('Running Code in Trial Mode to see if everything works properly...')
    raw_datasets = load_dataset('json', data_files=os.path.abspath(f'{args.dataset_location}/train.jsonl'),
                                split=['train[:160]', 'train[:10]'])
    train_dataset = raw_datasets[0].map(prepare_train_features, batched=True,
                                        remove_columns=raw_datasets[0].column_names)
    validation_dataset = raw_datasets[1].map(prepare_validation_features, batched=True,
                                             remove_columns=raw_datasets[1].column_names)
else:
    train_dataset_raw = load_dataset('json', data_files=os.path.abspath(f'{args.dataset_location}/train.jsonl'))
    train_dataset = train_dataset_raw['train'].map(prepare_train_features, batched=True,
                                                   remove_columns=train_dataset_raw['train'].column_names)

    validation_dataset_raw = load_dataset('json', data_files=os.path.abspath(f'{args.dataset_location}/dev.jsonl'))
    validation_dataset_raw = DatasetDict({"validation": validation_dataset_raw["train"]})
    validation_dataset = validation_dataset_raw['validation'].map(prepare_validation_features, batched=True,
                                                                  remove_columns=validation_dataset_raw[
                                                                      'validation'].column_names)

metric = load("squad_v2")  # Using v2 of squad since dataset contains impossible questions.

train_dataset.set_format("torch")
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size,
                              worker_init_fn=seed_worker, generator=g)

validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")
eval_dataloader = DataLoader(validation_set, collate_fn=data_collator, batch_size=batch_size,
                             worker_init_fn=seed_worker, generator=g)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
output_dir = args.trained_model_name

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
        metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets[1])
    else:
        metrics = compute_metrics(start_logits, end_logits, validation_dataset, validation_dataset_raw['validation'])

    print(f"epoch {epoch}:", metrics)

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        # repo.push_to_hub(commit_message=f"Training in progress epoch {epoch}", blocking=False)
