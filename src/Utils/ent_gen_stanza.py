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
                                                            'radiology'], 'tokenize':'default'})

dataset = load_dataset("Saptarshi7/covid_qa_cleaned_CS", use_auth_token=True)

new_ents = []
for i in tqdm(range(dataset['train'].num_rows)):
    doc = nlp(dataset['train'][i]['context'])
    for ent_dict in doc.entities:
        new_ents.append(ent_dict.text)

with open('stanza_ents-from_context.pkl', 'wb') as f:
    pickle.dump(new_ents, f)