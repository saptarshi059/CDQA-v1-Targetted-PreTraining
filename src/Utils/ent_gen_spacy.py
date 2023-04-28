from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
import pickle5 as pickle
from tqdm import tqdm
import argparse
import scispacy
import spacy
import re


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


def ent_gen(dataset_name, op_file, location, do_filter, local_dataset_format):
    print('Loading spacy model...')
    spacy.prefer_gpu(gpu_id=0)
    nlp = spacy.load("en_core_sci_sm")

    if location == 'remote':
        dataset = load_dataset(dataset_name)
    else:
        if local_dataset_format == 'parquet':
            dataset = load_dataset('parquet', data_files=dataset_name)
        else:
            dataset = load_dataset('../../data/RadQA'
                                   '/radqa-a-question-answering-dataset-to-improve-comprehension'
                                   '-of-radiology-reports-1.0.0'
                                   '/radqa.py')

    all_contexts = list(set(dataset['train']['context']))
    all_questions = list(set(dataset['train']['question']))

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

    if do_filter:
        print('Running filter on extracted entities...')

        total_ents = list(set(ctx_ents).union(set(ques_ents)))
        total_ents_filtered = []

        for ent in total_ents:
            if (not ent.isnumeric()) and (len(ent) > 5) and not (re.search(r'https*|doi:|\[\d+\]|[\u0080-\uFFFF]|et '
                                                                           r'al.|[Aa]uthor|www|\n|\*|\||@|;|&|!',
                                                                           ent)) \
                    and not (re.search(r'Fig[ures]*|\btables*\b|supplementary', ent, re.IGNORECASE)) \
                    and (ent.count('(') == ent.count(')')) \
                    and (ent.count('"') % 2 == 0) \
                    and (ent.count("\'") % 2 == 0) \
                    and (ent.count('[') == ent.count(']')):
                total_ents_filtered.append(ent)

        print(f'Total entities after filtering: {len(total_ents_filtered)}')

        total_ents_filtered.sort()
        vocab = {v: k for k, v in dict(enumerate(total_ents_filtered)).items()}

        vectorizer = TfidfVectorizer(vocabulary=vocab, stop_words='english', lowercase=False)
        x = vectorizer.fit_transform(all_contexts + all_questions)

        idf = vectorizer.idf_
        top_n_ents = []
        for k, _ in {k: v for k, v in sorted(dict(zip(vectorizer.get_feature_names(), idf)).items(),
                                             key=lambda item: item[1],
                                             reverse=True)}.items():
            top_n_ents.append(k)
            if len(top_n_ents) == 25000:
                break

    print('Saving entities...')
    with open(op_file, 'wb') as f:
        pickle.dump(list(set(total_ents)), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Saptarshi7/covid_qa_cleaned_CS', type=str)
    parser.add_argument('--dataset_location', default='remote', type=str)
    parser.add_argument('--local_dataset_format', default='parquet', type=str)
    parser.add_argument('--do_filter', default=False, type=str2bool)
    parser.add_argument('--output_file_name', default='ents_spacy.pkl', type=str)
    args = parser.parse_args()
    ent_gen(args.dataset, args.output_file_name, args.dataset_location, args.do_filter, args.local_dataset_format)
