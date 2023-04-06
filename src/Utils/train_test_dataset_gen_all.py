from datasets import load_dataset
from ent_gen_spacy import ent_gen
import pickle5 as pickle
import argparse


def main(dataset_name, corpus_file):
    dataset = load_dataset(dataset_name)

    print('Generating 80/20 split of dataset...')
    train_subset = dataset['train'].select(range(0, 1676))
    test_subset = dataset['train'].select(range(1676, len(dataset['train'])))

    print(f'Training Subset has: {len(train_subset)} samples ({len(train_subset) / len(dataset["train"])})%')
    print(f'Test Subset has: {len(test_subset)} samples ({len(test_subset) / len(dataset["train"])})%')
    print(f'Number of overlapping contexts '
          f'b/w train & test: {len(set(train_subset["context"]).intersection(set(test_subset["context"])))}')

    dataset_name = dataset_name.replace('/', '_')

    print('Saving subsets to disk...')
    train_subset.save_to_disk(f'{dataset_name}_train_subset')
    test_subset.save_to_disk(f'{dataset_name}_test_subset')

    print('Generating entities...')
    ent_gen(f'{dataset_name}_train_subset', f'{dataset_name}_train_subset_ents_spacy.pkl', 'local')

    with open(f'{dataset_name}_train_subset_ents_spacy.pkl', 'rb') as f:
        train_subset_ents = pickle.load(f)

    mlm_dataset = load_dataset("parquet", data_files=corpus_file).filter(input_columns='entity',
                                                                         function=lambda x: x in train_subset_ents)

    print(f'Saving MLM subset with ({len(mlm_dataset["train"])}) contexts...')
    mlm_dataset['train'].to_parquet(f"galactica_corpus_subset.parquet")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Saptarshi7/covid_qa_cleaned_CS', type=str)
    parser.add_argument('--generated_corpus_file', default='../../data/COVID-QA/generated_corpus_47k_scispacy_sm'
                                                           '/agg_gens.parquet',
                        type=str)
    args = parser.parse_args()
    main(args.dataset, args.generated_corpus_file)
