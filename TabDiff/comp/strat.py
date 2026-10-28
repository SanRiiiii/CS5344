"""Stratified, attack-avoiding selection from a synthetic pool.

Normal traffic has modes that overlap attack signatures (UNSW: sttl 254 / 62). A global
"lowest attack probability" filter drops those modes entirely, so their real test normals become
false positives. Here each mode gets a quota = m * its real share, and within a mode the rows with
the lowest classifier attack probability are kept; the remaining quota goes to the other rows.

    python comp/strat.py --dataname unsw --pool submissions/pool/unsw_e8000_pool5x.csv --m 0 0.25 0.5 1
"""
import argparse
import os

import numpy as np
import pandas as pd

from evaluate import anomaly_clusters, evaluate, load
from experiments import finish
from lco import fit_classifier

SEED = 0
MODES = {'unsw': ('sttl', [254, 62])}


def select(pool, p, real, meta, name, m, rng):
    col, values = MODES[name]
    n = meta['n_submit']
    picks, used = [], 0
    for v in values:
        quota = int(round(n * m * (real[col] == v).mean()))
        idx = np.flatnonzero(pool[col].to_numpy() == v)
        assert len(idx) >= quota, f'{col}={v}: pool has {len(idx)}, quota {quota}'
        picks.append(idx[np.argsort(p[idx])[:quota]])
        used += quota
    rest = np.flatnonzero(~pool[col].isin(values).to_numpy())
    picks.append(rest[np.argsort(p[rest])[:n - used]])
    return pool.iloc[np.concatenate(picks)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', required=True)
    parser.add_argument('--pool', required=True)
    parser.add_argument('--m', type=float, nargs='+', required=True)
    parser.add_argument('--save', default=None, help='write the submission for the (single) m')
    args = parser.parse_args()

    meta, holdout, anomalies, raw = load(args.dataname)
    meta['anomaly_rate'] = len(anomalies) / len(raw)
    clusters = anomaly_clusters(args.dataname, meta, anomalies)
    real = pd.read_csv(f'data/{args.dataname}/{args.dataname}.csv')
    for c, v in meta['const_cols'].items():
        real[c] = v
    real = real[meta['feature_cols']]
    pool = pd.read_csv(args.pool)[meta['feature_cols']]
    p = fit_classifier(real, anomalies, meta)(pool)

    for m in args.m:
        chosen = select(pool, p, real, meta, args.dataname, m, np.random.default_rng(SEED))
        sub = finish(chosen, real, meta)
        overall, cl = evaluate(sub, meta, holdout, anomalies, clusters)
        cl_auroc = [v[3] for v in cl.values()]
        col = MODES[args.dataname][0]
        row = {'dataname': args.dataname, 'pool': os.path.basename(args.pool), 'm': m,
               'score': overall[:, 0].mean(), 'iforest': overall[:, 1].mean(), 'ecod': overall[:, 2].mean(),
               'cl_mean': np.mean(cl_auroc), 'cl_worst': np.min(cl_auroc),
               'max_p_mode': float(p[chosen.index[chosen[col].isin(MODES[args.dataname][1])]].max()) if m > 0 else 0.0}
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        path = 'experiments/strat.csv'
        pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        if args.save:
            sub.to_csv(args.save, index=False)
            print('saved', args.save, sub.shape)


if __name__ == '__main__':
    main()
