import random

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
from evaluate import load
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from composer.utils import dist
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, default_data_collator, get_scheduler, set_seed
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

logging.set_verbosity(50)

# Initializing all random number methods.
g = torch.Generator()
g.manual_seed(42)
torch.manual_seed(42)
random.seed(42)
set_seed(42)


def _auto_wrap_policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
    if recurse:
        return True
    if hasattr(module, '_fsdp_wrap'):
        return bool(module._fsdp_wrap)
    return False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# train_dataset_raw = load_dataset('json', data_files='train.jsonl')
# dev_dataset_raw = DatasetDict({'validation': load_dataset('json', data_files='dev.jsonl')['train']})

train_dataset_raw = DatasetDict({'train': load_dataset('json', data_files='../../../data/RadQA/radqa-a-question'
                                                                          '-answering-dataset-to-improve'
                                                                          '-comprehension-of-radiology-reports-1.0.0'
                                                                          '/train.jsonl')['train'].select([0])})
dev_dataset_raw = DatasetDict({'validation': load_dataset('json', data_files='../../../data/RadQA/radqa-a-question'
                                                                             '-answering-dataset-to-improve'
                                                                             '-comprehension-of-radiology-reports-1.0.0'
                                                                             '/dev.jsonl')['train'].select([0])})

dist.initialize_dist(None)

model_checkpoint = "facebook/galactica-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, return_dict=True)

tokenizer.bos_token = '<s>'
tokenizer.pad_token = '<pad>'
tokenizer.eos_token = '</s>'


def encodeCLM(examples):
    samples = []
    for q, c in zip(examples['question'], examples['context']):
        samples.append(f'question: {q} context: {c} answer: ')
    input_text_tokenized = tokenizer(samples, return_tensors='pt', padding='max_length', max_length=1800,
                                     truncation=True)

    answers = [x['text'][0] if x['text'] != [] else 'Ġ' for x in examples['answers']]
    answers_tokenized = tokenizer(answers, return_tensors='pt', padding='max_length', max_length=1800, truncation=True)

    return {'input_ids': input_text_tokenized['input_ids'],
            'attention_mask': input_text_tokenized['attention_mask'],
            'labels': answers_tokenized['input_ids']}


metric = load("squad")
theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in dev_dataset_raw['validation']]


def compute_metrics(pred_tensors):
    decoded_preds = tokenizer.batch_decode(pred_tensors, skip_special_tokens=True)

    predicted_answers = []
    for sample, pred_text in zip(theoretical_answers, decoded_preds):
        predicted_answers.append({"id": sample["id"], "prediction_text": pred_text})

    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


batch_size = 1

accelerator = Accelerator()
# device = accelerator.device

#model._fsdp_wrap = True
model = accelerator.prepare(model)

# FSDP wrap
#fsdp_wrapped_gal = FSDP(model, use_orig_params=False)

data_collator = default_data_collator

train_dataset = train_dataset_raw['train'].map(encodeCLM, remove_columns=train_dataset_raw['train'].column_names,
                                               batched=True)
validation_dataset = dev_dataset_raw['validation'].map(encodeCLM,
                                                       remove_columns=dev_dataset_raw['validation'].column_names,
                                                       batched=True)

train_dataset.set_format("torch")
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size,
                              worker_init_fn=seed_worker, generator=g)
validation_dataset.set_format("torch")
eval_dataloader = DataLoader(validation_dataset, collate_fn=data_collator, batch_size=batch_size,
                             worker_init_fn=seed_worker, generator=g)

optimizer = AdamW(model.parameters(), lr=2e-5)

# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader,
                                                                                 eval_dataloader, lr_scheduler)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in tqdm(enumerate(train_dataloader)):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    accelerator.print("Evaluation!")
    predicted_tensors = []
    for step, batch in tqdm(enumerate(eval_dataloader)):
        with torch.no_grad():
        #    fsdp_wrapped_gal.forward(input_ids=batch['input_ids'])
        #with FSDP.summon_full_params(model, recurse=False):
            predicted_tensors.extend(model.generate(input_ids=batch['input_ids'],
                                                               attention_mask=batch['attention_mask'],
                                                               max_new_tokens=30))

    metrics = compute_metrics(predicted_tensors)

    print(f"epoch {epoch}:", metrics)
