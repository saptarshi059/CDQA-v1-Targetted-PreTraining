#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pickle5 as pickle

#selected_stanza_ents = pd.read_csv('../../data/our-wikipedia-corpus/Tokens_From_Question_side/mini_corpus-10T5CpT.csv')
stanza_ents_main = pd.read_pickle(open('../../data/stanza_ents-from_context.pkl','rb'))

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model_vocab = list(tokenizer.vocab.keys())


# In[ ]:


from tqdm.auto import tqdm

#Removing those entities that are also in the model vocab since we don't want 2 of the same token embedding.
ent_in_model_vocab = []
for ent in tqdm(stanza_ents_main):
    if ent in model_vocab:
        ent_in_model_vocab.append(ent)

for ent in ent_in_model_vocab:
    stanza_ents_main.remove(ent)

#Removing all occurrences of the selected entities. This is where we want fresh new entities & contexts.
'''
for ent in list(set(selected_stanza_ents.ent.to_list())):
    if ent in stanza_ents_main:
        stanza_ents_main = list(filter((ent).__ne__, stanza_ents_main))
'''

# In[ ]:


from collections import Counter
s = Counter(stanza_ents_main)
sorted_ent_counts = dict(sorted(s.items(), key=lambda item: item[1], reverse=True))


# In[ ]:


#Removing 1 length entities
keys_to_remove = []
for key in sorted_ent_counts.keys():
    if len(key) == 1:
        keys_to_remove.append(key)
        
for key in keys_to_remove:
    sorted_ent_counts.pop(key)

# In[1]:


import wikipedia
from collections import defaultdict

no_of_ents_to_select = 10000
no_of_results_per_entity = 1

selected_ents_text_dict = defaultdict(list)

for ent in tqdm(sorted_ent_counts.keys()):
    if len(selected_ents_text_dict) == no_of_ents_to_select:
        break
    
    search_results = wikipedia.search(str(ent), results=no_of_results_per_entity)
    
    no_of_ctx_available = 0
    final_ctx = ''
    
    for res in search_results:
        try:
            final_ctx += wikipedia.page(res, auto_suggest=False).content
            no_of_ctx_available += 1          
        except:
            continue
    
    if no_of_ctx_available == no_of_results_per_entity:
        answer = wikipedia.page(search_results[0], auto_suggest=False).summary
        if answer != '':
            selected_ents_text_dict[ent].append(answer)
            selected_ents_text_dict[ent].append(final_ctx)
        else:
            selected_ents_text_dict.pop(ent, None)

#print(f'Entities that were selected: {selected_ents_text_dict.keys()}')


# In[ ]:


sample_id = []
title = []
context = []
question = []
answers = []

for idx, (ent, page_data) in enumerate(selected_ents_text_dict.items()):
    sample_id.append(str(idx))
    title.append(ent)
    context.append(page_data[1])
    question.append(f'What is {ent}?')
    answers.append({'text': [page_data[0]], 'answer_start': [page_data[1].find(page_data[0])]})
    
#Saving the dataframe as parquet since it was messing up formats.
pd.DataFrame(zip(sample_id, title, context, question, answers), 
             columns = ['id', 'title', 'context', 'question'
                       ,'answers']).to_parquet('mini-corpus-for-extended-QA')