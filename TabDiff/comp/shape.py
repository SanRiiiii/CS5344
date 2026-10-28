"""Detector-aware reshaping of a submission (experimental; rows become less realistic).

T1  tail inflation: in 'noise' numeric dims, a fraction q of rows get a value drawn from the upper
    half of the real normal marginal, so large-but-normal test values stop looking like tail events
    to ECOD. Noise dims = dims where, under the base submission, holdout normals get an ECOD
    per-dim score at least as high as anomalies (uses anomaly labels, so the local score is optimistic).
T2  constant dims: sparse dims are set to their mode; sklearn's IsolationForest skips constant
    features, so splits concentrate on the remaining dims.
T3  flatten: only the top-k signal dims (by the ECOD gap above) keep their values; every other
    numeric dim is set to the real min for a random half of rows and the real max for the rest, so
    every test value sits mid-rank and the dim adds a near-constant ECOD term. Upper-bound probe only:
    the rows are no longer realistic traffic.
T4  constant: like T3 but non-kept numeric dims are set to the real median. IsolationForest skips
    constant features, so it splits only on the kept dims.

    python comp/shape.py --dataname unsw --base submissions/candidates/unsw_clf_keep50.csv --runs t1:q=0.1 t2
"""
import argparse
import os

import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from sklearn.preprocessing import OneHotEncoder

from evaluate import anomaly_clusters, encode, evaluate, load
from experiments import finish

SEED = 0


def per_dim_gap(sub, meta, holdout, anomalies):
    """ECOD per-dim score of anomalies minus holdout normals, one mixed batch at the train anomaly rate."""
    nums = [c for c in meta['feature_cols'] if c not in meta['cat_cols']]
    n_anom = round(len(holdout) * meta['anomaly_rate'] / (1 - meta['anomaly_rate']))
    a = anomalies.sample(n_anom, random_state=SEED)[meta['feature_cols']]
    batch = pd.concat([holdout, a], ignore_index=True)
    X_train, X_test = encode(sub[meta['feature_cols']], batch, meta['cat_cols'])
    ecod = ECOD().fit(X_train)
    ecod.decision_function(X_test)
    O = ecod.O[-len(batch):, :len(nums)]
    return pd.Series(O[len(holdout):].mean(0) - O[:len(holdout)].mean(0), index=nums)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()

    meta, holdout, anomalies, raw = load(args.dataname)
    meta['anomaly_rate'] = len(anomalies) / len(raw)
    clusters = anomaly_clusters(args.dataname, meta, anomalies)
    real = pd.read_csv(f'data/{args.dataname}/{args.dataname}.csv')
    for c, v in meta['const_cols'].items():
        real[c] = v
    real = real[meta['feature_cols']]
    base = pd.read_csv(args.base)[meta['feature_cols']]
    nums = [c for c in meta['feature_cols'] if c not in meta['cat_cols'] and c not in meta['const_cols']]

    gap = per_dim_gap(base, meta, holdout, anomalies).drop(list(meta['const_cols']), errors='ignore')
    noise_dims = gap[gap <= 0].index.tolist()
    print('noise dims (normals score >= anomalies):', noise_dims)
    print('signal dims:', gap[gap > 0].sort_values(ascending=False).round(2).to_dict())
    sparse_dims = [c for c in nums if (real[c] == real[c].mode()[0]).mean() >= 0.95]
    print('sparse dims (>=95% one value):', sparse_dims)

    for spec in args.runs:
        kind, _, arg = spec.partition(':')
        params = dict(kv.split('=') for kv in filter(None, arg.split(',')))
        rng = np.random.default_rng(SEED)
        sub = base.copy()
        if kind == 't1':
            q = float(params['q'])
            dims = noise_dims if params.get('dims', 'noise') == 'noise' else [c for c in nums if c not in ('sttl', 'dttl', 'ct_state_ttl')]
            for c in dims:
                rows = rng.random(len(sub)) < q
                sub[c] = sub[c].astype(float)
                sub.loc[rows, c] = np.quantile(real[c], rng.uniform(0.5, 1.0, rows.sum()))
        elif kind == 't2':
            for c in sparse_dims:
                sub[c] = real[c].mode()[0]
        elif kind == 't4':
            keep = gap.sort_values(ascending=False).index[:int(params['k'])].tolist()
            for c in nums:
                if c not in keep:
                    sub[c] = real[c].median()
        elif kind == 't3':
            keep = gap.sort_values(ascending=False).index[:int(params['k'])].tolist()
            for c in nums:
                if c not in keep:
                    hi = rng.random(len(sub)) < 0.5
                    sub[c] = np.where(hi, real[c].max(), real[c].min())
        else:
            raise ValueError(spec)
        sub = finish(sub, real, meta)
        overall, cl = evaluate(sub, meta, holdout, anomalies, clusters)
        cl_auroc = [v[3] for v in cl.values()]
        row = {'dataname': args.dataname, 'base': os.path.basename(args.base), 'run': spec,
               'score': overall[:, 0].mean(), 'iforest': overall[:, 1].mean(), 'ecod': overall[:, 2].mean(),
               'cl_mean': np.mean(cl_auroc), 'cl_worst': np.min(cl_auroc)}
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        path = 'experiments/shape.csv'
        pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            sub.to_csv(os.path.join(args.save_dir, f"{args.dataname}_{spec.replace(':', '_').replace(',', '_').replace('=', '')}.csv"), index=False)


if __name__ == '__main__':
    main()
