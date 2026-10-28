"""How hard to filter: keep the most normal-looking share of a synthetic pool.

For each keep fraction f, a random subset of n_submit / f pool rows is drawn and the n_submit rows
with the lowest classifier attack probability are kept. The classifier is trained on all anomalies,
which is slightly optimistic locally (comp/lco.py measures the leave-cluster-out version).

    python comp/filter_pool.py --dataname unsw --pool submissions/pool/unsw_e8000_pool5x.csv --keep 0.5 0.33 0.2
    python comp/filter_pool.py ... --keep 0.2 --save submissions/candidates/unsw_keep20.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from evaluate import anomaly_clusters, evaluate, load
from experiments import finish
from lco import fit_classifier

SEED = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', required=True)
    parser.add_argument('--pool', required=True)
    parser.add_argument('--keep', type=float, nargs='+', required=True)
    parser.add_argument('--save', default=None, help='write the submission for the (single) keep fraction')
    args = parser.parse_args()

    meta, holdout, anomalies, raw = load(args.dataname)
    meta['anomaly_rate'] = len(anomalies) / len(raw)
    clusters = anomaly_clusters(args.dataname, meta, anomalies)
    real = pd.read_csv(f'data/{args.dataname}/{args.dataname}.csv')
    for c, v in meta['const_cols'].items():
        real[c] = v
    real = real[meta['feature_cols']]
    pool = pd.read_csv(args.pool)[meta['feature_cols']]
    n = meta['n_submit']
    p = fit_classifier(real, anomalies, meta)(pool)

    rows = []
    for f in args.keep:
        m = round(n / f)
        assert m <= len(pool), f'keep={f} needs {m} pool rows, pool has {len(pool)}'
        sub_idx = np.random.default_rng(SEED).choice(len(pool), m, replace=False)
        keep = sub_idx[np.argsort(p[sub_idx])[:n]]
        sub = finish(pool.iloc[keep], real, meta)
        overall, cl = evaluate(sub, meta, holdout, anomalies, clusters)
        cl_auroc = [v[3] for v in cl.values()]
        row = {'keep': f, 'score': overall[:, 0].mean(), 'iforest': overall[:, 1].mean(), 'ecod': overall[:, 2].mean(),
               'cl_mean': np.mean(cl_auroc), 'cl_worst': np.min(cl_auroc), 'max_p_kept': p[keep].max(),
               **({'sttl254%': (sub['sttl'] == 254).mean() * 100} if 'sttl' in sub else {})}
        rows.append(row)
        print({k: round(v, 4) for k, v in row.items()}, flush=True)
        if args.save:
            os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
            sub.to_csv(args.save, index=False)
            print('saved', args.save, sub.shape)
    out = pd.DataFrame(rows)
    out.insert(0, 'pool', os.path.basename(args.pool))
    out.insert(0, 'dataname', args.dataname)
    path = 'experiments/filter_pool.csv'
    out.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


if __name__ == '__main__':
    main()
