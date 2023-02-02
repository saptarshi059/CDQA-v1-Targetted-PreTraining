import sqlite3
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
import re
from pymetamap import MetaMap
import os
from time import sleep

def load_metamap():
	
	# Setup UMLS Server
	metamap_base_dir = os.path.abspath('../../../metamap_experiments/public_mm/')
	metamap_bin_dir = os.path.relpath('bin/metamap20')
	metamap_pos_server_dir = os.path.relpath('bin/skrmedpostctl')
	metamap_wsd_server_dir = os.path.relpath('bin/wsdserverctl')

	# Start servers
	os.system(os.path.join(metamap_base_dir, metamap_pos_server_dir) + ' start') # Part of speech tagger
	os.system(os.path.join(metamap_base_dir, metamap_wsd_server_dir) + ' start') # Word sense disambiguation 

	# Sleep a bit to give time for these servers to start up
	sleep(60)

	metamap_instance = MetaMap.get_instance(os.path.join(metamap_base_dir, metamap_bin_dir))

	return metamap_instance

def main(metam):
	dataset = load_dataset("Saptarshi7/covid_qa_cleaned_CS", use_auth_token=True)

	all_questions = dataset['train']['question']
	all_contexts = list(set(dataset['train']['context']))

	total_covidqa_ctx_and_ques = all_questions + all_contexts

	'''
	total_covidqa_text = ''
	for ques in tqdm(all_questions):
		total_covidqa_text = total_covidqa_text + ' ' + ques

	for ctx in tqdm(all_contexts):
		total_covidqa_text = total_covidqa_text + ' ' + ctx
	'''

	def get_keys_from_mm(concept, klist):
		conc_dict = concept._asdict()
		conc_list = [conc_dict.get(kk) for kk in klist]
		return(tuple(conc_list))

	for item in tqdm(total_covidqa_ctx_and_ques):
		cons, errs = metam.extract_concepts([item],
                                word_sense_disambiguation = True,
                                composite_phrase = 1, # for memory issues
                                prune = 30)

		keys_of_interest = ['cui', 'preferred_name']
		cols = [get_keys_from_mm(cc, keys_of_interest) for cc in cons]
		results_df = pd.DataFrame(cols, columns = keys_of_interest)

	print('Saving extracted concepts...')
	results_df.to_csv('extracted_concepts.csv', index=False)

if __name__ == '__main__':
	metam = load_metamap()
	main(metam)