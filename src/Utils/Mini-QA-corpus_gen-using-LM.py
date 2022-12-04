#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle5 as pickle
import os
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from collections import Counter
from transformers import pipeline
import pandas as pd

pred_file_path = os.path.abspath('../../data/stanza_ents-from_questions.pkl')
stanza_ents_main = pickle.load(open(pred_file_path, 'rb'))


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model_vocab = list(tokenizer.vocab.keys())

#Removing those entities that are also in the model vocab since we don't want 2 of the same token embedding.
ent_in_model_vocab = []
for ent in tqdm(stanza_ents_main):
    if ent in model_vocab:
        ent_in_model_vocab.append(ent)

for ent in ent_in_model_vocab:
    stanza_ents_main.remove(ent)


# In[ ]:


#Removing 1 length entities
keys_to_remove = []
for key in stanza_ents_main:
    if len(key) == 1:
        keys_to_remove.append(key)
        
for key in keys_to_remove:
    stanza_ents_main.remove(key)

#Keeping only unique enitities
stanza_ents_main = list(set(stanza_ents_main))


# In[ ]:


#Setting up the text-generator
generator = pipeline('text-generation', model = "bigscience/bloom-560m", device=0)


# In[ ]:


sample_id = []
title = []
contexts = []
questions = []
answers = []

def triple_gen(ent, ques):
    questions.append(ques)
    ctx_body = generator(ent, renormalize_logits=True, 
                              do_sample=True, max_new_tokens=2000)[0]['generated_text']
    
    ans_prefix = ques[ques.find(',')+1: ques.find('?')].strip()
    ans_text = generator(ans_prefix, renormalize_logits=True, do_sample=True,
                         max_new_tokens=100)[0]['generated_text']
    
    total_context = ctx_body + '' + ans_text
    contexts.append(total_context)
    
    answers.append({'text': [ans_text], 'answer_start': [total_context.find(ans_text)]})

c = 0
for ent in tqdm(stanza_ents_main):
    #Question-1 (Is)
    sample_id.append(str(c))
    c += 1
    title.append('Is' + ent)
    triple_gen(ent, generator(f"Question: Is {ent},", renormalize_logits=True, do_sample=True, 
                               forced_eos_token_id=[34], 
                               max_new_tokens=10)[0]['generated_text'].strip('Question: '))
    
    #Question-2 (In)
    sample_id.append(str(c))
    c += 1
    title.append('In' + ent)
    triple_gen(ent, generator(f"Question: In {ent},", renormalize_logits=True, do_sample=True, 
                               forced_eos_token_id=[34], 
                               max_new_tokens=10)[0]['generated_text'].strip('Question: '))
    
    #Question-3 (Why)
    sample_id.append(str(c))
    c += 1
    title.append('Why' + ent)
    triple_gen(ent, generator(f"Question: Why {ent},", renormalize_logits=True, do_sample=True, 
                               forced_eos_token_id=[34], 
                               max_new_tokens=10)[0]['generated_text'].strip('Question: '))
    
    #Question-4 (During)
    sample_id.append(str(c))
    c += 1
    title.append('During' + ent)
    triple_gen(ent, generator(f"Question: During {ent},", renormalize_logits=True, do_sample=True, 
                               forced_eos_token_id=[34], 
                               max_new_tokens=10)[0]['generated_text'].strip('Question: '))
    
    #Question-5 (Which)
    sample_id.append(str(c))
    c += 1
    title.append('Which' + ent)
    triple_gen(ent, generator(f"Question: Which {ent},", renormalize_logits=True, do_sample=True, 
                               forced_eos_token_id=[34], 
                               max_new_tokens=10)[0]['generated_text'].strip('Question: '))
    
#Saving the dataframe as parquet since it was messing up formats.
pd.DataFrame(zip(sample_id, title, context, question, answers), 
             columns = ['id', 'title', 'context', 'question'
                       ,'answers']).to_parquet('mini-corpus-for-extended-QA')

