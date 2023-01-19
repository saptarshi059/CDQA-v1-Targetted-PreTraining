__author__ = 'Connor Heaton'

import os
import argparse

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../../out/gen_v1')

    args = parser.parse_args()

    data_files = [
        os.path.join(args.data_dir, fp) for fp in os.listdir(args.data_dir)
        if 'agg' not in fp and fp.endswith('parquet')
    ]
    print('Found {} generation files...'.format(len(data_files)))

    print('Reading and concatenating...')
    dfs = [pd.read_parquet(fp) for fp in data_files]
    agg_df = pd.concat(dfs, axis=0)
    print('agg_df: {}'.format(agg_df))
    agg_df.to_parquet(
        os.path.join(args.data_dir, 'agg_gens.parquet')
    )
    print('Done :)')


