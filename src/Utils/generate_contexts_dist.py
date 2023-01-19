__author__ = 'Connor Heaton'

import re
import os
import math
import time
import argparse

import pandas as pd
import pickle5 as pickle

from tqdm.auto import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_model', default="facebook/galactica-1.3b", type=str)
    parser.add_argument('--student_model', default="distilbert-base-uncased", type=str)
    parser.add_argument('--no_new_question_tokens', default=20, type=int)
    parser.add_argument('--no_new_context_tokens', default=2048, type=int)
    parser.add_argument('--no_new_answer_tokens', default=512, type=int)
    parser.add_argument('--last_N_tokens_for_context', default=80, type=int)

    parser.add_argument('--n_context_per_entity', default=5, type=int)

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int, help='zero-indexed')
    parser.add_argument('--out', default='../../out/gen_v1')
    parser.add_argument('--summary_every', default=1, type=int)

    parser.add_argument('--batch_size', default=-1, type=int)

    args = parser.parse_args()

    if args.batch_size > 0:
        args.batch_size = args.n_context_per_entity

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    out_fp = os.path.join(args.out, 'rank{}_gens.parquet'.format(args.rank))

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    model_vocab = list(tokenizer.vocab.keys())

    print('[rank {}] Reading entities'.format(args.rank))
    stanza_ents_file_path = os.path.abspath('../../data/stanza_ents-from_questions.pkl')
    with open(stanza_ents_file_path, 'rb') as f:
        stanza_ents_main = pickle.load(f)

    # Keeping only unique enitities
    stanza_ents_main = list(sorted(set(stanza_ents_main)))
    # stanza_ents_main = stanza_ents_main[:2]
    print('stanza_ents_main[:10]: {}'.format(stanza_ents_main[:10]))

    n_ents = len(stanza_ents_main)
    ents_per_rank = int(math.ceil(n_ents / args.world_size))
    rank_ents = stanza_ents_main[args.rank * ents_per_rank: (args.rank + 1) * ents_per_rank]
    print('[rank {}] n_ents: {}'.format(args.rank, n_ents))
    print('[rank {}] ents_per_rank: {}'.format(args.rank, ents_per_rank))
    print('[rank {}] len(rank_ents): {}'.format(args.rank, len(rank_ents)))

    print('[rank {}] Making pipeline...'.format(args.rank, len(rank_ents)))
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    generator_model = AutoModelForCausalLM.from_pretrained(args.teacher_model)
    generator_model_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    generator_model_question_mark_ID = generator_model_tokenizer.get_vocab()['?']

    generator = pipeline('text-generation', model=generator_model, tokenizer=generator_model_tokenizer,
                         device=args.rank)
    generator.tokenizer.pad_token_id = generator_model.config.eos_token_id

    print('[rank {}] Beginning to iterate over entities...'.format(args.rank))
    write_data = []
    start_time = time.time()
    for entity_idx, entity in enumerate(rank_ents):
        prompts = ['Title: {}'.format(entity) for _ in range(args.n_context_per_entity)]
        # print('len(prompts): {}'.format(len(prompts)))

        set_seed(42)
        generations = generator(
            prompts, renormalize_logits=True, do_sample=True, max_new_tokens=args.no_new_answer_tokens,
            top_p=0.9, temperature=0.9, use_cache=True, batch_size=args.batch_size
        )
        generations = [gen[0]['generated_text'] for gen in generations]
        # print('\n*********************************\n'.join(generations))

        for generation in generations:
            write_data.append(
                [entity_idx, 'Title: {}'.format(entity), generation]
            )

        if entity_idx % args.summary_every == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_entity = elapsed_time / (entity_idx + 1)
            avg_time_per_context = elapsed_time / (len(write_data))
            print_str = '[rank {0}] Generated contexts for {1}/{2} entities - {3:.2f}s/entity {4:.2f}s/context'.format(
                args.rank, entity_idx + 1, len(rank_ents), avg_time_per_entity, avg_time_per_context
            )
            print(print_str)

    print('[rank {}] Saving data... out_fp={}'.format(args.rank, out_fp))
    entity_idxs, titles, generations = map(list, zip(*write_data))
    write_df = pd.DataFrame(zip(entity_idxs, titles, generations),
                            columns=['id', 'title', 'context'])
    # print(write_df)
    write_df.to_parquet(out_fp)
    print('[rank {}] Done :)'.format(args.rank))


