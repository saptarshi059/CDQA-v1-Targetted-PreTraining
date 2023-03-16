import stanza
from tqdm.auto import tqdm
from datasets import load_dataset
import pickle5 as pickle

nlp = stanza.Pipeline('en', package=None, processors={'ner':['anatem',
                                                            'bc5cdr',
                                                            'bc4chemd',
                                                            'bionlp13cg',
                                                            'jnlpba',
                                                            'linnaeus',
                                                            'ncbi_disease',
                                                            's800',
                                                            'i2b2',
                                                            'radiology'], 'tokenize':'default'}, use_gpu=True)

dataset = load_dataset("Saptarshi7/covid_qa_cleaned_CS", use_auth_token=True)

new_ents = []

all_questions = dataset['train']['question']
all_contexts = list(set(dataset['train']['context']))

for ques in tqdm(all_questions):
  doc = nlp(ques)
  for ent_dict in doc.entities:
    new_ents.append(ent_dict.text)

for ctx in tqdm(all_contexts):
  doc = nlp(ctx)
  for ent_dict in doc.entities:
    new_ents.append(ent_dict.text)

print(f'Total Entities: {len(new_ents)} | Unique Entities: {len(list(set(new_ents)))}')

with open('stanza_ents-covidqa.pkl', 'wb') as f:
    pickle.dump(new_ents, f)