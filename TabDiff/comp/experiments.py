"""Hypothesis experiments: alternative generators and resampling schemes, scored with comp/evaluate.py.

Every generator only sees the real 90% training normals (never the holdout or any anomaly label)
and returns a full submission. All runs share the same anomaly resamples, so differences between
runs are paired. Results are appended to experiments/results.csv.

    python comp/experiments.py --dataname unsw --runs real_bootstrap noise_q0.1 smote_k5
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from evaluate import anomaly_clusters, evaluate, load

SEED = 0
VOLUME_COLS = {
    'unsw': ['spkts', 'dpkts', 'sbytes', 'dbytes', 'sloss', 'dloss'],
    'nslkdd': ['src_bytes', 'dst_bytes'],
}


# ---------------------------------------------------------------- helpers

def finish(df, real, meta):
    """Clip to the real range, round integer columns, restore column order and add the id column."""
    df = df.copy()
    for c in meta['feature_cols']:
        if c in meta['cat_cols'] or c in meta['const_cols']:
            continue
        df[c] = df[c].clip(real[c].min(), real[c].max())
        if c in meta['int_cols']:
            df[c] = df[c].round().astype(np.int64)
    df = df[meta['feature_cols']].reset_index(drop=True)
    df.insert(0, 'id', np.arange(len(df)))
    return df


def num_cols(meta):
    return [c for c in meta['feature_cols'] if c not in meta['cat_cols'] and c not in meta['const_cols']]


def gauss_space(real, meta):
    """Map numeric columns to a standard normal scale via the empirical quantiles."""
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', subsample=10**9, random_state=SEED)
    qt.fit(real[num_cols(meta)])
    return qt


def resample(real, n, weights=None, rng=None):
    p = None if weights is None else weights / weights.sum()
    idx = rng.choice(len(real), n, replace=True, p=p)
    return real.iloc[idx].reset_index(drop=True)


# ---------------------------------------------------------------- generators
# Each takes (real, meta, name, rng, **params) and returns a submission dataframe.

def gen_real_bootstrap(real, meta, name, rng, weights=None):
    """Reference only (copies real rows): resample real normals, optionally weighted."""
    return finish(resample(real, meta['n_submit'], weights, rng), real, meta)


def gen_noise_q(real, meta, name, rng, scale=0.1, weights=None):
    """Bootstrap + Gaussian noise of `scale` std in quantile-normal space; categoricals kept."""
    base = resample(real, meta['n_submit'], weights, rng)
    qt, cols = gauss_space(real, meta), num_cols(meta)
    z = qt.transform(base[cols]) + rng.normal(0, scale, (len(base), len(cols)))
    base[cols] = qt.inverse_transform(z)
    return finish(base, real, meta)


def gen_smote(real, meta, name, rng, k=5, weights=None):
    """Interpolate each sampled row towards one of its k nearest real neighbours (same categories)."""
    qt, cols, cats = gauss_space(real, meta), num_cols(meta), meta['cat_cols']
    Z = qt.transform(real[cols])
    out = []
    n_total = meta['n_submit']
    key = real[cats].astype(str).agg('|'.join, axis=1)
    base_idx = rng.choice(len(real), n_total, replace=True, p=None if weights is None else weights / weights.sum())
    base_key = key.to_numpy()[base_idx]
    for kv in np.unique(base_key):
        members = np.flatnonzero(key.to_numpy() == kv)
        rows = base_idx[base_key == kv]
        if len(members) < 2:
            out.append(real.iloc[rows])
            continue
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(members))).fit(Z[members])
        _, nb = nn.kneighbors(Z[rows])
        pick = members[nb[np.arange(len(rows)), rng.integers(1, nb.shape[1], len(rows))]]
        lam = rng.random((len(rows), 1))
        zz = Z[rows] + lam * (Z[pick] - Z[rows])
        part = real.iloc[rows].copy()
        part[cols] = qt.inverse_transform(zz)
        out.append(part)
    return finish(pd.concat(out).sample(frac=1, random_state=SEED), real, meta)


def gen_file(real, meta, name, rng, path=None):
    return pd.read_csv(path)


def gen_pool(real, meta, name, rng, path=None, weights_fn=None):
    """Draw the submission without replacement from a larger synthetic pool, with row weights computed on the pool."""
    pool = pd.read_csv(path)[meta['feature_cols']]
    w = weights_fn(pool) if weights_fn else np.ones(len(pool))
    idx = rng.choice(len(pool), meta['n_submit'], replace=False, p=w / w.sum())
    return finish(pool.iloc[idx], real, meta)


# ---------------------------------------------------------------- weighting schemes (H2 / H3)

def volume_weights(real, name, alpha, top=0.2):
    """Upweight rows whose total volume (rank-averaged over volume columns) is in the top `top` share."""
    r = real[VOLUME_COLS[name]].rank(pct=True).mean(axis=1)
    return np.where(r >= r.quantile(1 - top), alpha, 1.0)


def sttl_weights(real, name, alpha):
    return np.where(real['sttl'] == 254, alpha, 1.0) if 'sttl' in real else np.ones(len(real))


GENERATORS = {'real_bootstrap': gen_real_bootstrap, 'noise_q': gen_noise_q, 'smote': gen_smote, 'file': gen_file,
              'pool': gen_pool}


def parse_run(spec, real, name):
    """'noise_q:scale=0.2,vol=2' -> (generator, params). vol / sttl turn into row weights."""
    gen, _, arg = spec.partition(':')
    params = {}
    for kv in filter(None, arg.split(',')):
        k, v = kv.split('=')
        params[k] = v if k == 'path' else float(v)
    vol, top, sttl = params.pop('vol', None), params.pop('top', 0.2), params.pop('sttl', None)

    def weights_fn(df):
        w = np.ones(len(df))
        if vol is not None:
            w = w * volume_weights(df, name, vol, top)
        if sttl is not None:
            w = w * sttl_weights(df, name, sttl)
        return w

    if gen == 'pool':
        params['weights_fn'] = weights_fn
    elif gen != 'file' and (vol is not None or sttl is not None):
        params['weights'] = weights_fn(real)
    if 'k' in params:
        params['k'] = int(params['k'])
    return GENERATORS[gen], params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--save_dir', default=None, help='also write each generated submission here')
    args = parser.parse_args()

    meta, holdout, anomalies, raw = load(args.dataname)
    meta['anomaly_rate'] = len(anomalies) / len(raw)
    clusters = anomaly_clusters(args.dataname, meta, anomalies)
    real = pd.read_csv(f'data/{args.dataname}/{args.dataname}.csv')
    for c, v in meta['const_cols'].items():
        real[c] = v
    real = real[meta['feature_cols']]

    os.makedirs('experiments', exist_ok=True)
    out_path = 'experiments/results.csv'
    for spec in args.runs:
        t = time.time()
        gen, params = parse_run(spec, real, args.dataname)
        sub = gen(real, meta, args.dataname, np.random.default_rng(SEED), **params)
        assert len(sub) == meta['n_submit'] and list(sub.columns) == ['id'] + meta['feature_cols'], spec
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            fname = spec.replace('pool:path=', '').replace('/', '_').replace(':', '_').replace(',', '_').replace('=', '')
            sub.to_csv(os.path.join(args.save_dir, f'{args.dataname}__{fname}.csv'), index=False)
        overall, per_cluster = evaluate(sub, meta, holdout, anomalies, clusters)
        cl = {f'c{k}': v[3] for k, v in per_cluster.items()}
        row = {'dataname': args.dataname, 'run': spec,
               'score': overall[:, 0].mean(), 'iforest': overall[:, 1].mean(), 'ecod': overall[:, 2].mean(),
               'auroc': overall[:, 3].mean(), 'cl_mean': np.mean(list(cl.values())), 'cl_worst': min(cl.values()),
               **cl, 'secs': round(time.time() - t)}
        pd.DataFrame([row]).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
        print(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()), flush=True)


if __name__ == '__main__':
    main()
