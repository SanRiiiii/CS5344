"""Search how much of each kind of synthetic normal goes into the submission, by the competition metric.

The synthetic pool is clustered into K groups. A candidate is a weight per group; the submission is
filled group by group in proportion to weight * group size (capped by group size), taking a fixed
random order inside each group so the objective is deterministic. CMA-ES maximises the fast proxy
of the leaderboard score, (mean IForest AUPRC over 3 seeds + ECOD AUPRC) / 2.

With --linear, each row's log-weight is its group weight plus a linear function of its scaled
features, and the n_submit rows are taken by Gumbel top-k with fixed noise (a deterministic weighted
draw without replacement), so selection can also move along feature directions inside groups.

Overfitting guard: holdout normals and anomalies are split in half. The search only sees the 'opt'
half; the 'val' half is reported alongside and decides which candidate is kept. The classifier used
as a clustering feature is trained on opt anomalies only.

    python comp/optimize_selection.py --dataname unsw --pool submissions/pool/unsw_e8000_pool5x.csv \
        --k 40 --evals 360 --save submissions/candidates/unsw_opt.csv
"""
import argparse
import json
import os
import time

import cma
import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from evaluate import anomaly_clusters, evaluate, load
from experiments import finish
from lco import fit_classifier

SEED = 0
PROXY_SEEDS = (0, 1, 2)
N_BATCHES = 2


class Objective:
    def __init__(self, pool, groups, meta, batches, Z=None):
        self.meta, self.groups, self.batches = meta, groups, batches
        self.Z = Z                                   # scaled features for the linear term, or None
        self.k = groups.max() + 1
        self.gumbel = -np.log(-np.log(np.random.default_rng(SEED + 7).random(len(pool))))
        cats = meta['cat_cols']
        nums = [c for c in meta['feature_cols'] if c not in cats]
        self.enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(pool[cats].astype(str))
        self.encode = lambda d: np.hstack([d[nums].to_numpy(float), self.enc.transform(d[cats].astype(str))])
        self.X_pool = self.encode(pool)
        self.X_batches = [(self.encode(b), y) for b, y in batches]
        rng = np.random.default_rng(SEED)
        self.members = [rng.permutation(np.flatnonzero(groups == g)) for g in range(groups.max() + 1)]
        self.sizes = np.array([len(m) for m in self.members])
        self.n = meta['n_submit']

    def allocate(self, theta):
        """Counts per group proportional to exp(theta) * size, capped by size; leftover is redistributed."""
        w = np.exp(np.clip(theta, -10, 10)) * self.sizes
        counts = np.zeros(len(w), dtype=int)
        free = np.ones(len(w), dtype=bool)
        left = self.n
        while left > 0:
            share = w * free
            want = np.floor(left * share / share.sum()).astype(int)
            room = self.sizes - counts
            add = np.minimum(want, room)
            if add.sum() == 0:                       # rounding leftovers: give to the heaviest free group
                g = np.argmax(np.where(free & (room > 0), w, -1))
                add[g] = 1
            counts += add
            left -= add.sum()
            free &= counts < self.sizes
        return counts

    def rows(self, theta):
        theta = np.asarray(theta)
        if self.Z is None:
            counts = self.allocate(theta)
            return np.concatenate([m[:c] for m, c in zip(self.members, counts)])
        logw = np.clip(theta[:self.k], -10, 10)[self.groups] + self.Z @ theta[self.k:]
        return np.argpartition(-(logw + self.gumbel), self.n)[:self.n]

    def score(self, idx, which):
        X_train = self.X_pool[idx]
        ecod = ECOD().fit(X_train)
        forests = [IForest(random_state=s).fit(X_train) for s in PROXY_SEEDS]
        out = []
        for X_b, y in [self.X_batches[i] for i in which]:
            e = average_precision_score(y, ecod.decision_function(X_b))
            i = np.mean([average_precision_score(y, f.decision_function(X_b)) for f in forests])
            out.append((e + i) / 2)
        return float(np.mean(out))


def split_batches(meta, holdout, anomalies, rate, clusters=None):
    """N_BATCHES opt batches and N_BATCHES val batches from disjoint halves of normals and anomalies.

    With clusters given, anomalies are split by pseudo attack type instead (even clusters -> opt,
    odd clusters -> val), so 'val' measures attack types the search never saw."""
    rng = np.random.default_rng(SEED)
    h = rng.permutation(len(holdout))
    if clusters is None:
        a = rng.permutation(len(anomalies))
        opt_a, val_a = anomalies.iloc[a[: len(a) // 2]], anomalies.iloc[a[len(a) // 2:]]
    else:
        even = (clusters.to_numpy() % 2 == 0)
        opt_a, val_a = anomalies[even], anomalies[~even]
    halves = {'opt': (holdout.iloc[h[: len(h) // 2]], opt_a),
              'val': (holdout.iloc[h[len(h) // 2:]], val_a)}
    batches, which = [], {'opt': [], 'val': []}
    for name, (norm, anom) in halves.items():
        n_anom = round(len(norm) * rate / (1 - rate))
        for b in range(N_BATCHES):
            s = anom.sample(min(n_anom, len(anom)), random_state=b)[meta['feature_cols']]
            batches.append((pd.concat([norm, s], ignore_index=True), np.r_[np.zeros(len(norm)), np.ones(len(s))]))
            which[name].append(len(batches) - 1)
    return batches, which, halves['opt'][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', required=True)
    parser.add_argument('--pool', nargs='+', required=True)
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--split', choices=['random', 'clusters'], default='random',
                        help='how anomalies are split between search and validation')
    parser.add_argument('--proxy_seeds', type=int, default=3)
    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--evals', type=int, default=360)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--no_source', action='store_true', help='do not use the pool index as a clustering feature (older runs)')
    parser.add_argument('--init_theta', default=None, help='warm start from a saved theta (.npy) of the same k/pool/linear setup')
    parser.add_argument('--save', default=None)
    args = parser.parse_args()

    meta, holdout, anomalies, raw = load(args.dataname)
    rate = len(anomalies) / len(raw)
    meta['anomaly_rate'] = rate
    real = pd.read_csv(f'data/{args.dataname}/{args.dataname}.csv')
    for c, v in meta['const_cols'].items():
        real[c] = v
    real = real[meta['feature_cols']]
    read = lambda f: pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f)
    parts = [read(f)[meta['feature_cols']] for f in args.pool]
    source = np.concatenate([np.full(len(d), i) for i, d in enumerate(parts)])
    pool = pd.concat(parts, ignore_index=True)
    global PROXY_SEEDS
    PROXY_SEEDS = tuple(range(args.proxy_seeds))
    clusters = anomaly_clusters(args.dataname, meta, anomalies)
    batches, which, opt_anomalies = split_batches(meta, holdout, anomalies, rate,
                                                  clusters if args.split == 'clusters' else None)

    # cluster the pool on scaled features plus the (opt-only) attack probability
    t = time.time()
    p = fit_classifier(real, opt_anomalies, meta)(pool)
    cats = meta['cat_cols']
    nums = [c for c in meta['feature_cols'] if c not in cats and c not in meta['const_cols']]
    Z = np.hstack([QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=SEED).fit_transform(pool[nums]),
                   OneHotEncoder(sparse_output=False).fit_transform(pool[cats].astype(str)),
                   3 * np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None] / 6,
                   3 * np.eye(len(parts))[source][:, : 0 if args.no_source else None]])    # keep generators apart
    groups = MiniBatchKMeans(args.k, random_state=SEED, n_init=5, batch_size=8192).fit_predict(Z)
    print(f'pool {len(pool)} rows -> {args.k} groups in {time.time() - t:.0f}s; sizes {np.bincount(groups).min()}..{np.bincount(groups).max()}', flush=True)

    Zlin = (Z - Z.mean(0)) / (Z.std(0) + 1e-9) / np.sqrt(Z.shape[1]) if args.linear else None
    obj = Objective(pool, groups, meta, batches, Zlin)
    start = np.zeros(args.k + (Z.shape[1] if args.linear else 0))
    if args.init_theta:
        start = np.load(args.init_theta)
        assert start.shape == (args.k + (Z.shape[1] if args.linear else 0),), 'init_theta does not match this setup'
    base_opt, base_val = obj.score(obj.rows(start), which['opt']), obj.score(obj.rows(start), which['val'])
    print(f"{'start' if args.init_theta else 'uniform pool'}: opt={base_opt:.4f} val={base_val:.4f}", flush=True)

    es = cma.CMAEvolutionStrategy(start, args.sigma, {'seed': SEED + 1, 'maxfevals': args.evals, 'verbose': -9})
    best = {'val': base_val, 'opt': base_opt, 'theta': start}
    history = []
    while not es.stop():
        cands = es.ask()
        fits = [-obj.score(obj.rows(np.asarray(c)), which['opt']) for c in cands]
        es.tell(cands, fits)
        mean = es.mean
        o, v = obj.score(obj.rows(mean), which['opt']), obj.score(obj.rows(mean), which['val'])
        history.append({'evals': es.countevals, 'best_opt_in_gen': -min(fits), 'mean_opt': o, 'mean_val': v})
        if v > best['val']:
            best = {'val': v, 'opt': o, 'theta': np.array(mean)}
        print(f"evals={es.countevals:4d} gen_best_opt={-min(fits):.4f} mean_opt={o:.4f} mean_val={v:.4f} best_val={best['val']:.4f}", flush=True)

    os.makedirs('experiments', exist_ok=True)
    tag = f"{args.dataname}_k{args.k}{'_lin' if args.linear else ''}_n{len(pool)}_{args.split}{'_warm' if args.init_theta else ''}"
    pd.DataFrame(history).to_csv(f'experiments/optsel_{tag}_history.csv', index=False)
    np.save(f'experiments/optsel_{tag}_theta.npy', best['theta'])
    np.save(f'experiments/optsel_{tag}_groups.npy', groups)

    idx = obj.rows(best['theta'])
    sub = finish(pool.iloc[idx], real, meta)
    counts = np.bincount(groups[idx], minlength=args.k)
    prof = pd.DataFrame({'size': obj.sizes, 'taken': counts, 'weight': np.exp(np.clip(best['theta'][:args.k], -10, 10)).round(3),
                         'mean_p': [p[groups == g].mean() for g in range(args.k)],
                         'source': [np.bincount(source[groups == g], minlength=len(parts)).argmax() for g in range(args.k)],
                         **({'sttl254%': [(pool.sttl[groups == g] == 254).mean() * 100 for g in range(args.k)]} if 'sttl' in pool else {})})
    pd.set_option('display.width', 200)
    print(prof.sort_values('taken', ascending=False).round(3).to_string())
    print('rows taken per source:', dict(zip(args.pool, np.bincount(source[idx], minlength=len(parts)))))
    print(f"chosen: opt={best['opt']:.4f} val={best['val']:.4f} (uniform: opt={base_opt:.4f} val={base_val:.4f})")

    overall, cl = evaluate(sub, meta, holdout, anomalies, clusters)
    print(f'full local evaluate (sees opt half too, optimistic): score={overall[:, 0].mean():.4f} '
          f'iforest={overall[:, 1].mean():.4f} ecod={overall[:, 2].mean():.4f} '
          f'cl_mean={np.mean([v[3] for v in cl.values()]):.4f} cl_worst={np.min([v[3] for v in cl.values()]):.4f}')
    print('AUROC per anomaly cluster (odd clusters are the unseen ones under --split clusters):',
          {f'c{k}': round(v[3], 4) for k, v in cl.items()})
    if args.save:
        sub.to_csv(args.save, index=False)
        print('saved', args.save, sub.shape)


if __name__ == '__main__':
    main()
