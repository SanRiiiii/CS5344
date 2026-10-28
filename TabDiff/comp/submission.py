"""Sample a trained TabDiff model and write a competition submission csv."""
import json
import os
import random

import numpy as np
import pandas as pd
import torch

OVERSAMPLE = 1.05   # draw a few extra rows so that rows with NaNs can be dropped


def to_submission(syn_df, meta, n_rows=None):
    """Restore dropped constant columns and the original column order, then prepend an id column."""
    n_rows = n_rows or meta['n_submit']
    df = syn_df.copy()
    for c, v in meta['const_cols'].items():
        df[c] = v
    df = df[meta['feature_cols']]
    df = df.dropna().head(n_rows).reset_index(drop=True)
    # Columns that only hold integers in the real data come out of the model as floats
    for c in meta.get('int_cols', []):
        df[c] = df[c].round().astype(np.int64)
    assert len(df) == n_rows, f"only {len(df)} valid rows, need {n_rows}"
    df.insert(0, 'id', np.arange(len(df)))
    return df


def write_submission(trainer, data_dir, submission_path, seed, pool_factor=1.0):
    """pool_factor > 1 writes a larger pool of rows (same format) for later filtering."""
    with open(os.path.join(data_dir, 'comp_meta.json')) as f:
        meta = json.load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n_rows = int(meta['n_submit'] * pool_factor)
    syn_df = trainer.sample_synthetic(int(n_rows * OVERSAMPLE))
    sub = to_submission(syn_df, meta, n_rows)

    if submission_path is None:
        submission_path = os.path.join(trainer.result_save_path, f'submission_seed{seed}.csv')
    os.makedirs(os.path.dirname(os.path.abspath(submission_path)), exist_ok=True)
    sub.to_csv(submission_path, index=False)
    print(f'Submission with shape {sub.shape} saved at {submission_path}')
