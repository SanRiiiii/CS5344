"""Turn the competition CSVs into TabDiff datasets (normal rows only).

Run from the TabDiff repo root:
    python comp/prepare_data.py
    python process_dataset.py --dataname unsw
    python process_dataset.py --dataname nslkdd

For each dataset this writes
    data/<name>/<name>.csv      normal rows used for training (90%)
    data/<name>/holdout.csv     held-out normal rows (10%), TabDiff's "test" split
    data/<name>/eval_set.csv    holdout normals + anomalies sampled to train.csv's anomaly rate, for local scoring
    data/<name>/comp_meta.json  original column order, dropped constant columns, submission size
    data/Info/<name>.json       TabDiff dataset info
"""
import json
import os

import numpy as np
import pandas as pd

COMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATASETS = {
    'unsw': {
        'raw': 'UNSW-NB15.csv',
        'cat_cols': ['proto', 'service', 'state'],
        'n_submit': 70000,
    },
    'nslkdd': {
        'raw': 'NSL-KDD.csv',
        'cat_cols': ['protocol_type', 'service', 'flag'],
        'n_submit': 40000,
    },
}
LABEL = 'is_anomaly'
HOLDOUT_FRAC = 0.1
SEED = 42


def split_keep_categories(df, cat_cols, frac, seed):
    """Random split where every category seen in the holdout also appears in train."""
    rng = np.random.default_rng(seed)
    is_hold = rng.random(len(df)) < frac
    train, hold = df[~is_hold], df[is_hold]
    unseen = np.zeros(len(hold), dtype=bool)
    for c in cat_cols:
        unseen |= ~hold[c].isin(set(train[c])).to_numpy()
    train = pd.concat([train, hold[unseen]])
    hold = hold[~unseen]
    return train.reset_index(drop=True), hold.reset_index(drop=True)


def prepare(name, cfg):
    raw = pd.read_csv(os.path.join(COMP_DIR, cfg['raw']))
    feature_cols = [c for c in raw.columns if c != LABEL]

    normal = raw[raw[LABEL] == 0][feature_cols]
    anomaly = raw[raw[LABEL] == 1]

    # Constant columns carry no signal and break quantile scaling; re-added at export time.
    const_cols = {c: normal[c].iloc[0].item() for c in feature_cols if normal[c].nunique() == 1}
    model_cols = [c for c in feature_cols if c not in const_cols]
    cat_cols = [c for c in cfg['cat_cols'] if c in model_cols]
    num_cols = [c for c in model_cols if c not in cat_cols]

    # TabDiff needs one target column; use the last categorical one as a binclass target.
    # It is modelled jointly with the other categoricals, so this only affects bookkeeping.
    target_col = cat_cols[-1]
    ordered = num_cols + cat_cols[:-1] + [target_col]

    train, hold = split_keep_categories(normal[ordered], cat_cols, HOLDOUT_FRAC, SEED)

    out_dir = os.path.join('data', name)
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, f'{name}.csv'), index=False)
    hold.to_csv(os.path.join(out_dir, 'holdout.csv'), index=False)

    # AUPRC depends on the anomaly rate, so keep the eval set at the same rate as train.csv
    rate = len(anomaly) / len(raw)
    n_anom = round(len(hold) * rate / (1 - rate))
    anomaly = anomaly.sample(n=n_anom, random_state=SEED)
    eval_set = pd.concat([hold.assign(**{LABEL: 0}), anomaly[ordered + [LABEL]]], ignore_index=True)
    for c, v in const_cols.items():
        eval_set[c] = v
    eval_set[feature_cols + [LABEL]].to_csv(os.path.join(out_dir, 'eval_set.csv'), index=False)

    with open(os.path.join(out_dir, 'comp_meta.json'), 'w') as f:
        json.dump({
            'feature_cols': feature_cols,
            'cat_cols': cfg['cat_cols'],
            'const_cols': const_cols,
            'int_cols': [c for c in num_cols if (normal[c] % 1 == 0).all()],
            # rows of the raw csv used as eval_set anomalies; anything trained on labels must skip them
            'eval_anomaly_rows': anomaly.index.tolist(),
            'n_submit': cfg['n_submit'],
        }, f, indent=4)

    n_num = len(num_cols)
    info = {
        'name': name,
        'task_type': 'binclass',
        'header': 'infer',
        'column_names': None,
        'num_col_idx': list(range(n_num)),
        'cat_col_idx': list(range(n_num, len(ordered) - 1)),
        'target_col_idx': [len(ordered) - 1],
        'file_type': 'csv',
        'data_path': f'data/{name}/{name}.csv',
        'val_path': None,
        'test_path': f'data/{name}/holdout.csv',
    }
    os.makedirs('data/Info', exist_ok=True)
    with open(f'data/Info/{name}.json', 'w') as f:
        json.dump(info, f, indent=4)

    print(f'{name}: train {train.shape}, holdout {hold.shape}, eval_set {eval_set.shape}, '
          f'num {len(num_cols)}, cat {len(cat_cols)}, dropped const {list(const_cols)}')


if __name__ == '__main__':
    for name, cfg in DATASETS.items():
        prepare(name, cfg)
