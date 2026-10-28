"""Local proxy for the leaderboard score.

Trains ECOD and IForest (10 seeds) on a submission csv and scores data/<name>/eval_set.csv,
which holds the held-out normal rows plus anomalies sampled to train.csv's anomaly rate (1/3).
Score = (mean IForest AUPRC + ECOD AUPRC) / 2, as on Kaggle.

    python comp/score.py --dataname nslkdd --submission path/to/submission.csv
    python comp/score.py --dataname nslkdd --real      # reference: train on the real normal rows
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder

IFOREST_SEEDS = range(10)


def encode(train, test, cat_cols):
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    enc.fit(train[cat_cols].astype(str))
    def f(df):
        num = df.drop(columns=cat_cols).to_numpy(dtype=np.float64)
        return np.hstack([num, enc.transform(df[cat_cols].astype(str))])
    return f(train), f(test)


def score(train_df, eval_df, meta):
    feats, cat_cols = meta['feature_cols'], meta['cat_cols']
    X_train, X_test = encode(train_df[feats], eval_df[feats], cat_cols)
    y = eval_df['is_anomaly'].to_numpy()

    ecod = ECOD().fit(X_train)
    ecod_ap = average_precision_score(y, ecod.decision_function(X_test))

    if_aps = []
    for s in IFOREST_SEEDS:
        m = IForest(random_state=s).fit(X_train)
        if_aps.append(average_precision_score(y, m.decision_function(X_test)))
    if_ap = float(np.mean(if_aps))

    return {'score': (if_ap + ecod_ap) / 2, 'iforest': if_ap, 'iforest_std': float(np.std(if_aps)), 'ecod': ecod_ap}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', required=True)
    parser.add_argument('--submission', default=None)
    parser.add_argument('--real', action='store_true', help='score the real training normals instead')
    args = parser.parse_args()

    data_dir = f'data/{args.dataname}'
    with open(os.path.join(data_dir, 'comp_meta.json')) as f:
        meta = json.load(f)
    eval_df = pd.read_csv(os.path.join(data_dir, 'eval_set.csv'))

    if args.real:
        train_df = pd.read_csv(os.path.join(data_dir, f'{args.dataname}.csv'))
        for c, v in meta['const_cols'].items():
            train_df[c] = v
        name = 'real normals'
    else:
        train_df = pd.read_csv(args.submission)
        assert len(train_df) == meta['n_submit'], f"expected {meta['n_submit']} rows, got {len(train_df)}"
        assert list(train_df.columns) == ['id'] + meta['feature_cols'], 'column order does not match'
        assert train_df['id'].is_unique
        name = args.submission

    res = score(train_df, eval_df, meta)
    print(f"[{args.dataname}] {name}: score={res['score']:.4f}  "
          f"iforest={res['iforest']:.4f}±{res['iforest_std']:.4f}  ecod={res['ecod']:.4f}")


if __name__ == '__main__':
    main()
