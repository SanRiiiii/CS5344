"""Local evaluation that also asks how a submission copes with attack types it was not tuned on.

Detectors (ECOD + IForest over 10 seeds) are fitted once on the submission. IForest scores
holdout normals + every anomaly of the training file once and metrics are read off subsets.
pyod's ECOD builds its ECDFs from the training data *plus the batch being scored*, so a row's
ECOD score depends on what else is in the batch; ECOD therefore re-scores every subset as its
own batch, the way a hidden test set would be scored.

  overall   anomalies resampled REPEATS times at train.csv's anomaly rate; mean and std of the
            competition score (mean IForest AUPRC + ECOD AUPRC) / 2
  clusters  anomalies are grouped by KMeans into pseudo attack types (no attack-type labels exist);
            each cluster is scored against the holdout normals on its own, by AUROC because clusters
            are too small for a fixed anomaly rate and AUPRC moves with that rate. The macro mean and
            the worst cluster show whether a method only works for some kinds of attack.

Cluster assignments are cached in data/<name>/anomaly_clusters.csv, so that methods which use
anomaly labels can leave one cluster out and be tested on it.

    python comp/evaluate.py --dataname unsw --submission a.csv b.csv
    python comp/evaluate.py --dataname unsw --real            # reference: the real training normals
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

IFOREST_SEEDS = range(10)
REPEATS = 10
N_CLUSTERS = 8
SEED = 42
COMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW = {'unsw': 'UNSW-NB15.csv', 'nslkdd': 'NSL-KDD.csv'}


def load(name):
    data_dir = f'data/{name}'
    with open(os.path.join(data_dir, 'comp_meta.json')) as f:
        meta = json.load(f)
    raw = pd.read_csv(os.path.join(COMP_DIR, RAW[name]))
    anomalies = raw[raw['is_anomaly'] == 1]
    holdout = pd.read_csv(os.path.join(data_dir, 'holdout.csv'))
    for c, v in meta['const_cols'].items():
        holdout[c] = v
    return meta, holdout[meta['feature_cols']], anomalies, raw


def anomaly_clusters(name, meta, anomalies):
    """KMeans pseudo attack types, keyed by the anomaly's row index in the raw csv."""
    path = f'data/{name}/anomaly_clusters.csv'
    if os.path.exists(path):
        return pd.read_csv(path, index_col='raw_row')['cluster'].reindex(anomalies.index)
    cats = meta['cat_cols']
    nums = [c for c in meta['feature_cols'] if c not in cats]
    X = np.hstack([
        QuantileTransformer(n_quantiles=1000, random_state=SEED).fit_transform(anomalies[nums]),
        OneHotEncoder(sparse_output=False).fit_transform(anomalies[cats].astype(str)),
    ])
    labels = KMeans(N_CLUSTERS, n_init=10, random_state=SEED).fit_predict(X)
    clusters = pd.Series(labels, index=anomalies.index, name='cluster')
    clusters.rename_axis('raw_row').to_csv(path)
    return clusters


def describe_clusters(meta, anomalies, clusters, raw):
    normal = raw[raw['is_anomaly'] == 0]
    rows = []
    for k in sorted(clusters.unique()):
        a = anomalies[clusters == k]
        row = {'cluster': k, 'size': len(a), 'share%': round(len(a) / len(anomalies) * 100, 1)}
        for c in meta['cat_cols']:
            top = a[c].value_counts(normalize=True)
            row[c] = f'{top.index[0]} {top.iloc[0] * 100:.0f}%'
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"(normal rows for reference: {len(normal)}; "
          + ', '.join(f"{c} top = {normal[c].value_counts().index[0]}" for c in meta['cat_cols']) + ')')


def encode(train, test, cat_cols):
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    enc.fit(train[cat_cols].astype(str))
    def f(df):
        num = df.drop(columns=cat_cols).to_numpy(dtype=np.float64)
        return np.hstack([num, enc.transform(df[cat_cols].astype(str))])
    return f(train), f(test)


def fit_detectors(train_df, test_df, meta):
    """The fitted ECOD, the encoded test rows, and one IForest score vector per seed."""
    X_train, X_test = encode(train_df[meta['feature_cols']], test_df[meta['feature_cols']], meta['cat_cols'])
    ecod = ECOD().fit(X_train)
    iforest = [IForest(random_state=s).fit(X_train).decision_function(X_test) for s in IFOREST_SEEDS]
    return ecod, X_test, iforest


def metrics(y, ecod, X_test, iforest, idx):
    yy = y[idx]
    ecod_scores = ecod.decision_function(X_test[idx])    # scored as one batch, see module docstring
    e = average_precision_score(yy, ecod_scores)
    i = np.mean([average_precision_score(yy, s[idx]) for s in iforest])
    auroc = (roc_auc_score(yy, ecod_scores) + np.mean([roc_auc_score(yy, s[idx]) for s in iforest])) / 2
    return (e + i) / 2, i, e, auroc


def evaluate(train_df, meta, holdout, anomalies, clusters):
    test = pd.concat([holdout, anomalies[meta['feature_cols']]], ignore_index=True)
    y = np.r_[np.zeros(len(holdout)), np.ones(len(anomalies))]
    ecod, X_test, iforest = fit_detectors(train_df, test, meta)

    n_norm = len(holdout)
    normal_idx = np.arange(n_norm)
    anom_pos = np.arange(n_norm, len(test))
    rate = meta['anomaly_rate']
    n_anom = round(n_norm * rate / (1 - rate))
    rng = np.random.default_rng(SEED)

    overall = [metrics(y, ecod, X_test, iforest, np.r_[normal_idx, rng.choice(anom_pos, n_anom, replace=False)])
               for _ in range(REPEATS)]
    overall = np.array(overall)

    cl = clusters.to_numpy()
    per_cluster = {}
    for k in sorted(np.unique(cl)):
        pos = anom_pos[cl == k]
        take = min(len(pos), n_anom)
        reps = [metrics(y, ecod, X_test, iforest, np.r_[normal_idx, rng.choice(pos, take, replace=False)])
                for _ in range(REPEATS if take < len(pos) else 1)]
        per_cluster[k] = np.mean(reps, axis=0)
    return overall, per_cluster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', required=True)
    parser.add_argument('--submission', nargs='*', default=[])
    parser.add_argument('--real', action='store_true', help='also score the real training normals')
    parser.add_argument('--describe', action='store_true', help='print what each anomaly cluster looks like')
    args = parser.parse_args()

    meta, holdout, anomalies, raw = load(args.dataname)
    meta['anomaly_rate'] = len(anomalies) / len(raw)
    clusters = anomaly_clusters(args.dataname, meta, anomalies)
    if args.describe:
        describe_clusters(meta, anomalies, clusters, raw)

    runs = []
    if args.real:
        real = pd.read_csv(f'data/{args.dataname}/{args.dataname}.csv')
        for c, v in meta['const_cols'].items():
            real[c] = v
        runs.append(('real normals', real))
    for path in args.submission:
        df = pd.read_csv(path)
        assert len(df) == meta['n_submit'] and list(df.columns) == ['id'] + meta['feature_cols'], f'bad format: {path}'
        runs.append((os.path.basename(path), df))

    summary, cluster_table = [], {}
    for label, df in runs:
        overall, per_cluster = evaluate(df, meta, holdout, anomalies, clusters)
        cl_auroc = np.array([v[3] for v in per_cluster.values()])
        summary.append({
            'submission': label,
            'score': overall[:, 0].mean(), 'score_std': overall[:, 0].std(),
            'iforest': overall[:, 1].mean(), 'ecod': overall[:, 2].mean(), 'auroc': overall[:, 3].mean(),
            'cluster_auroc_mean': cl_auroc.mean(), 'cluster_auroc_worst': cl_auroc.min(),
        })
        cluster_table[label] = {k: v[3] for k, v in per_cluster.items()}

    print(f'\n[{args.dataname}] overall over {REPEATS} anomaly resamples, plus per-cluster score')
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    sizes = clusters.value_counts().sort_index()
    ct = pd.DataFrame(cluster_table)
    ct.index = [f'c{k} (n={sizes[k]})' for k in ct.index]
    print('\nAUROC (mean of ECOD and IForest) by anomaly cluster')
    print(ct.round(4).to_string())


if __name__ == '__main__':
    main()
