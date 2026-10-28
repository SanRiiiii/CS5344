
"""
CS5344 Starter Kit
Simple baseline augmentation for Kaggle submission.

Generate exactly TARGET_N synthetic NORMAL samples:
1. Keep is_anomaly == 0 rows.
2. Sample with replacement.
3. Add small Gaussian noise to numeric non-binary features.
4. Keep categorical and binary features unchanged.
5. Save id + feature columns as submission.csv.
"""

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

TRAIN_CSV = "train.csv"
OUTPUT_CSV = "sample_submission.csv"

LABEL_COLUMN = "is_anomaly"
TARGET_N = 2000
RANDOM_SEED = 42
JITTER_SCALE = 0.5


# ============================================================
# 1. Load data
# ============================================================

train = pd.read_csv(TRAIN_CSV)

if LABEL_COLUMN not in train.columns:
    raise ValueError(f"Training data must contain '{LABEL_COLUMN}'.")

feature_columns = [c for c in train.columns if c != LABEL_COLUMN]
normal = train.loc[train[LABEL_COLUMN] == 0, feature_columns].copy()

if len(normal) == 0:
    raise ValueError("No normal samples found.")

if normal.isna().any().any():
    raise ValueError("Normal training data contains missing values.")

print(f"Train shape: {train.shape}")
print(f"Normal rows: {len(normal)}")
print(f"Features: {len(feature_columns)}")


# ============================================================
# 2. Identify feature types
# ============================================================

categorical_columns = [
    c for c in feature_columns
    if pd.api.types.is_object_dtype(normal[c])
    or pd.api.types.is_string_dtype(normal[c])
    or isinstance(normal[c].dtype, pd.CategoricalDtype)
]

numeric_columns = [c for c in feature_columns if c not in categorical_columns]

binary_columns = []
integer_columns = []
continuous_columns = []

for col in numeric_columns:
    values = normal[col].dropna().to_numpy()
    unique_values = set(np.unique(values).tolist())

    if unique_values.issubset({0, 1}):
        binary_columns.append(col)
    elif np.all(np.isclose(values, np.round(values))):
        integer_columns.append(col)
    else:
        continuous_columns.append(col)

print(f"Numeric: {len(numeric_columns)}")
print(f"Categorical: {len(categorical_columns)}")
print(f"Binary: {len(binary_columns)}")
print(f"Integer: {len(integer_columns)}")
print(f"Continuous: {len(continuous_columns)}")


# ============================================================
# 3. Bootstrap normal samples
# ============================================================

N_PROTOTYPES = 20

prototype_pool = normal.sample(
    n=N_PROTOTYPES,
    random_state=RANDOM_SEED,
)

synthetic = prototype_pool.sample(
    n=TARGET_N,
    replace=True,
    random_state=RANDOM_SEED,
).reset_index(drop=True)


# ============================================================
# 4. Add jitter
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

for col in integer_columns + continuous_columns:
    std = normal[col].std(ddof=0)

    if not np.isfinite(std) or std == 0:
        continue

    synthetic[col] = synthetic[col].astype(float) + rng.normal(
        0,
        JITTER_SCALE * std,
        TARGET_N,
    )

    lower, upper = normal[col].min(), normal[col].max()
    synthetic[col] = synthetic[col].clip(lower, upper)

    if col in integer_columns:
        synthetic[col] = np.rint(synthetic[col]).clip(lower, upper).astype(normal[col].dtype)


# ============================================================
# 5. Create Kaggle submission
# ============================================================

submission = synthetic[feature_columns].copy()
submission.insert(0, "id", np.arange(TARGET_N))


# ============================================================
# 6. Validate
# ============================================================

if len(submission) != TARGET_N:
    raise ValueError(f"Submission must contain exactly {TARGET_N} rows.")

if submission.columns.tolist() != ["id"] + feature_columns:
    raise ValueError("Submission columns are incorrect.")

if submission.isna().any().any():
    raise ValueError("Submission contains missing values.")

numeric_values = submission.select_dtypes(include=[np.number]).to_numpy(dtype=float)

if not np.isfinite(numeric_values).all():
    raise ValueError("Submission contains NaN or infinite values.")


# ============================================================
# 7. Save
# ============================================================

submission.to_csv(OUTPUT_CSV, index=False)

print("\nSubmission generated successfully.")
print(f"Output: {OUTPUT_CSV}")
print(f"Shape: {submission.shape}")
print(submission.head())