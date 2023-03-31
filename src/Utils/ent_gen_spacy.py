from datasets import load_dataset
from tqdm.auto import tqdm
import pickle5 as pickle
import scispacy
import spacy


def main():
    print('Loading spacy model...')
    spacy.prefer_gpu(gpu_id=0)
    nlp = spacy.load("en_core_sci_sm")

    dataset = load_dataset('Saptarshi7/covid_qa_cleaned_CS')

    all_contexts = list(set(dataset['train']['context']))
    all_questions = dataset['train']['question']

    ques_ents = []
    ctx_ents = []

    spacy.prefer_gpu(gpu_id=0)
    for ques in tqdm(all_questions):
        ques_ents.extend([str(x) for x in nlp(ques).ents])

    spacy.prefer_gpu(gpu_id=0)
    for ctx in tqdm(all_contexts):
        ctx_ents.extend([str(x) for x in nlp(ctx).ents])

    total_ents = ques_ents + ctx_ents
    print(f'Total Entities from Questions: {len(ques_ents)} (Unique: {len(set(ques_ents))})')
    print(f'Total Entities from Contexts: {len(ctx_ents)} (Unique: {len(set(ctx_ents))})')
    print(f'Total Entities from Questions + Contexts: {len(ques_ents + ctx_ents)} '
          f'(Unique: {len(set(ques_ents + ctx_ents))})')

    print('Saving entities...')
    with open('ents_spacy.pkl', 'wb') as f:
        pickle.dump(list(set(total_ents)), f)


if __name__ == '__main__':
    main()
