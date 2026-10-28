"""Classifier-guided resampling of a synthetic pool, checked against attack types the classifier never saw.

A gradient-boosted classifier (real training normals vs anomalies) estimates how attack-like every
pool row is; the submission is drawn from the pool with weights that shrink with that probability.

Leave-cluster-out: for each anomaly cluster k the classifier is trained without cluster k and the
submission is scored on cluster k only, so the gain on k is a gain on an attack type it never saw.
An 'all' classifier trained on every anomaly is also scored overall (optimistic, since the local
eval anomalies were then seen by the classifier).

    python comp/lco.py --dataname unsw --pool submissions/pool/unsw_e8000_pool2x.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

from evaluate import anomaly_clusters, evaluate, load
from experiments import finish

SEED = 0
VARIANTS = {
    'soft2': lambda p: (1 - p) ** 2,
    'soft8': lambda p: (1 - p) ** 8,
    'lowest': None,     # keep the n_submit rows with the lowest attack probability
}


def fit_classifier(normals, anomalies, meta):
    cats = meta['cat_cols']
    X = pd.concat([normals, anomalies])[meta['feature_cols']]
    y = np.r_[np.zeros(len(normals)), np.ones(len(anomalies))]
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(X[cats].astype(str))
    prep = lambda d: np.hstack([d[[c for c in meta['feature_cols'] if c not in cats]].to_numpy(float),
                                enc.transform(d[cats].astype(str))])
    n_num = len(meta['feature_cols']) - len(cats)
    clf = HistGradientBoostingClassifier(categorical_features=list(range(n_num, n_num + len(cats))),
                                         max_iter=300, learning_rate=0.1, random_state=SEED)
    clf.fit(prep(X), y)
    return lambda d: clf.predict_proba(prep(d))[:, 1]


def draw(pool, p, variant, n, rng):
    if VARIANTS[variant] is None:
        return pool.iloc[np.argsort(p)[:n]]
    w = VARIANTS[variant](p) + 1e-12
    return pool.iloc[rng.choice(len(pool), n, replace=False, p=w / w.sum())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', required=True)
    parser.add_argument('--pool', required=True)
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
    run = lambda sub: evaluate(finish(sub, real, meta), meta, holdout, anomalies, clusters)

    rng = np.random.default_rng(SEED)
    base_overall, base_cl = run(pool.iloc[rng.choice(len(pool), n, replace=False)])
    rows = [{'variant': 'uniform pool', 'trained_on': '-', 'score': base_overall[:, 0].mean(),
             **{f'c{k}': v[3] for k, v in base_cl.items()}}]

    # classifier on every anomaly: overall score (optimistic)
    p_all = fit_classifier(real, anomalies, meta)(pool)
    for v in VARIANTS:
        o, cl = run(draw(pool, p_all, v, n, np.random.default_rng(SEED)))
        rows.append({'variant': v, 'trained_on': 'all anomalies', 'score': o[:, 0].mean(),
                     **{f'c{k}': x[3] for k, x in cl.items()}})
        print(rows[-1], flush=True)

    # leave-cluster-out: AUROC on the held-out cluster only
    lco = {v: {} for v in VARIANTS}
    for k in sorted(clusters.unique()):
        p_k = fit_classifier(real, anomalies[clusters != k], meta)(pool)
        for v in VARIANTS:
            _, cl = run(draw(pool, p_k, v, n, np.random.default_rng(SEED)))
            lco[v][f'c{k}'] = cl[k][3]
        print(f'cluster {k} left out:', {v: round(lco[v][f"c{k}"], 4) for v in VARIANTS}, flush=True)
    for v in VARIANTS:
        rows.append({'variant': v, 'trained_on': 'all but the scored cluster', 'score': np.nan, **lco[v]})

    out = pd.DataFrame(rows)
    ccols = [c for c in out.columns if c.startswith('c')]
    out['cl_mean'] = out[ccols].mean(axis=1)
    out['cl_worst'] = out[ccols].min(axis=1)
    os.makedirs('experiments', exist_ok=True)
    out.insert(0, 'dataname', args.dataname)
    out.to_csv(f'experiments/lco_{args.dataname}.csv', index=False)
    pd.set_option('display.width', 250)
    print(out.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
