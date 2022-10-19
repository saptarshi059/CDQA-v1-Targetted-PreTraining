import wikipedia
import pandas as pd
from tqdm import tqdm

s = pd.read_csv('Filtered_by_Stanza_NER.csv')
s.drop(axis=1, inplace=True, columns=['Unnamed: 0'])
s.drop_duplicates(inplace=True)

texts = []
ents = []
unique_entities = list(set(s.natural_text.to_list()))

for ent in tqdm(unique_entities):
    try:
        texts.append(wikipedia.page(wikipedia.search(str(ent), results=1)).content)
        ents.append(str(ent))
    except:
        continue

pd.DataFrame(zip(ents, texts), columns = ['ent' ,'text']).to_csv('corpus.csv', index=False)