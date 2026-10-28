"""Write a large candidate pool from one of the cheap generators in comp/experiments.py.

    python comp/make_pool.py --dataname unsw --gen noise_q --scale 0.2 --factor 5 --out submissions/pool/unsw_noise02_pool5x.csv
    python comp/make_pool.py --dataname unsw --gen smote --k 5 --factor 5 --out submissions/pool/unsw_smote_pool5x.csv
"""
import argparse

import numpy as np
import pandas as pd

from evaluate import load
from experiments import gen_noise_q, gen_smote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', required=True)
    parser.add_argument('--gen', choices=['noise_q', 'smote'], required=True)
    parser.add_argument('--scale', type=float, default=0.2)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--factor', type=float, default=5)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    meta, _, _, _ = load(args.dataname)
    real = pd.read_csv(f'data/{args.dataname}/{args.dataname}.csv')
    for c, v in meta['const_cols'].items():
        real[c] = v
    real = real[meta['feature_cols']]
    meta = dict(meta, n_submit=int(meta['n_submit'] * args.factor))
    rng = np.random.default_rng(args.seed)
    if args.gen == 'noise_q':
        pool = gen_noise_q(real, meta, args.dataname, rng, scale=args.scale)
    else:
        pool = gen_smote(real, meta, args.dataname, rng, k=args.k)
    pool.to_csv(args.out, index=False)
    print('saved', args.out, pool.shape)


if __name__ == '__main__':
    main()
