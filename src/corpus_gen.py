import wikipedia
import pandas as pd
from tqdm import tqdm
import pickle5 as pickle

'''
s = pd.read_csv('Filtered_by_Stanza_NER.csv')
s.drop(axis=1, inplace=True, columns=['Unnamed: 0'])
s.drop_duplicates(inplace=True)
'''

stanza_ents = list(set(pickle.load(open('../data/stanza_ents.pkl', 'rb'))))
texts = []

'''
ents = []
unique_entities = list(set(s.natural_text.to_list()))
'''

for ent in tqdm(stanza_ents):
    try:
        texts.append(wikipedia.page(wikipedia.search(str(ent), results=1)).content)
        #ents.append(str(ent))
    except:
        continue

pd.DataFrame(zip(ents, texts), columns = ['ent' ,'text']).dropna().to_csv('corpus_only_stanza.csv', index=False)