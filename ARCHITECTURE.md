# AutoML Framework — Architecture & Technical Reference

---

## Architecture & Pipeline Overview

### System Flow

```
Dataset Upload
    → EDA Analysis (column types, missing values, outliers, cardinality, typo detection)
    → Adaptive Preprocessing (task detection, model-aware transforms, pipeline creation)
    → AutoML Training (Optuna HPO per model, stability checks, generalization scoring)
    → Model Selection (penalty-based ranking, rejection of overfit models)
    → Results (metrics, plots, saved model + pipeline)
```

### Core Components

| Component | Location | Role |
|---|---|---|
| `adaptive_preprocessing.py` | `backend/app/services/` | Context-aware preprocessing with EDA integration |
| `automl_engine.py` | `backend/app/services/` | Optuna-based training, generalization scoring, stability detection |
| `automl_plots.py` | `backend/app/services/` | Visualization generation (confusion matrix, ROC, residuals, etc.) |
| `eda_service.py` | `backend/app/services/` | Exploratory data analysis including typo detection |
| `automl.py` | `backend/app/routes/` | FastAPI endpoints for training and model management |
| `datasets.py` | `backend/app/routes/` | Dataset upload, preprocessing, EDA endpoints |
| `schemas.py` | `backend/app/models/` | Pydantic request/response schemas |

### Technology Stack

- **Backend**: FastAPI, Python 3.10+
- **ML**: scikit-learn 1.4.0, Optuna 3.5.0, XGBoost 2.0.3, LightGBM 4.3.0
- **Frontend**: React, Vite, TailwindCSS, Framer Motion
- **Storage**: JSON file storage (`backend/app/services/storage.py`, `data/` directory)
- **Visualization**: matplotlib 3.8.2, seaborn 0.13.1

### Prerequisites

- Python 3.10+
- Node.js 16+

### Setup

```
# Backend
cd backend && python -m venv venv && .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py  # serves on :8000

# Frontend
cd frontend && npm install && npm run dev  # serves on :5173
```

---

## Data Preprocessing Pipeline

The base preprocessing pipeline applies 10 sequential steps. This runs when the adaptive preprocessor is not engaged (e.g., standalone dataset preprocessing via the `/preprocess` endpoint).

### Step 1 — Load Dataset

Reads CSV with encoding fallback chain: UTF-8 → Latin-1 → ISO-8859-1 → CP1252 → UTF-16. Uses `chardet` as final fallback.

### Step 2 — Duplicate Removal

Drops exact duplicate rows.

### Step 3 — Constant Column Removal

Drops columns where `nunique() <= 1`.

### Step 4 — Missing Value Analysis

Columns with >50% missing values are flagged for removal.

### Step 5 — ID Column Detection

Pattern-based detection using exact matches:
- `col == 'id'`
- `col.endswith('_id')`
- `col.startswith('id_')`

Prevents false positives on words containing "id" (e.g., `Width`, `video`).

### Step 6 — Outlier Clipping (IQR)

For each numerical column:
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
x_clipped = clip(x, lower, upper)
```

### Step 7 — Missing Value Imputation

- **Numerical**: median
- **Categorical**: mode

### Step 8 — Categorical Encoding

| Condition | Strategy |
|---|---|
| Column is the target | Label encoding (category codes → 0, 1, 2, ...) |
| Feature with < 10 unique values | One-hot encoding (`drop_first=True`, `dtype=float64`) |
| Feature with >= 10 unique values | Label encoding |

`drop_first=True` avoids multicollinearity by encoding k categories into k-1 binary columns.

### Step 9 — Selective Scaling

`StandardScaler` (mean=0, std=1) applied to original numerical features only. Excluded from scaling:
- One-hot encoded columns (kept as 0/1)
- Target column

### Step 10 — Type Conversion

All `numpy` types converted to native Python types before JSON serialization.

### Configuration Defaults

```python
OUTLIER_IQR_MULTIPLIER = 1.5
HIGH_MISSING_THRESHOLD = 0.5    # >50% missing → drop column
LOW_CARDINALITY_THRESHOLD = 10  # <10 unique → one-hot encode
```

---

## Adaptive Preprocessing

The adaptive preprocessor replaces the base pipeline when `use_adaptive=true`. It uses EDA results and the target model family to select strategies dynamically.

### Task Type Detection

```python
if target_column is None:
    → UNSUPERVISED
elif target is numerical AND (unique_values / total > 0.05):
    → REGRESSION
else:
    → CLASSIFICATION
```

### Model Family Mapping

| Model | Family | Scaling | Encoding |
|---|---|---|---|
| Logistic Regression, Linear Regression, Ridge, Lasso, Elastic Net | LINEAR | StandardScaler | OneHotEncoder |
| Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM | TREE_BASED | None | OrdinalEncoder |
| KNN, SVM/SVR | DISTANCE_BASED | StandardScaler or RobustScaler | OneHotEncoder |
| Neural Network / MLP | NEURAL_NETWORK | MinMaxScaler | OneHotEncoder |

### Column Removal Logic

```
ID columns:
  - Name matches 'id', '*_id', 'id_*'   → DROP
  - Uniqueness ratio > 95%               → DROP
  - Sequential integers (1, 2, 3, ...)   → DROP

Constant columns: unique_values <= 1     → DROP
High missing: missing_ratio > 60%        → DROP
```

### Imputation Strategy

**Numerical features:**
```
if missing_ratio > 30% AND has_outliers → MEDIAN
elif missing_ratio > 30%               → MEAN
elif has_outliers                       → MEDIAN
else                                    → MEAN
```

**Categorical features:** always MOST_FREQUENT.

### Encoding Strategy

```
if model_family == TREE_BASED           → ORDINAL
elif unique_count > 50                  → ORDINAL (high cardinality)
else                                    → ONE_HOT
```

### Scaling Strategy

```
if model_family == TREE_BASED           → NONE
elif model_family == NEURAL_NETWORK     → MINMAX
elif has_significant_outliers           → ROBUST
else                                    → STANDARD
```

### Pipeline Structure (sklearn)

```
ColumnTransformer([
    ('num', numerical_pipeline, numerical_columns),
    ('cat', categorical_pipeline, categorical_columns),
    ('text', text_pipeline, text_columns)        # if text columns detected
])
```

### Quality Score

```python
quality_score = 100.0
quality_score -= (columns_removed / initial_columns) * 20  # max -20
quality_score -= (missing_cells / total_cells) * 30         # max -30
quality_score = clamp(0, quality_score, 100)
```

### Decision Log

Every decision is recorded as:
```json
{
    "timestamp": "ISO-8601",
    "category": "imputation | encoding | scaling | id_detection | ...",
    "decision": "what was done",
    "reason": "why",
    "impact": "info | warning | critical"
}
```

### Custom Overrides

Users can supply:
- `force_drop_columns` / `force_keep_columns`
- `imputation_overrides` (per-column strategy)
- `encoding_overrides` (per-column strategy)
- `disable_id_detection`
- `missing_threshold_override`

---

## Text Data Handling

### Text Column Detection Heuristics

A column is classified as text (not categorical) if:

```python
if (avg_length > 30 AND unique_count > 50 AND unique_ratio > 0.5) OR
   (avg_length > 50 AND (has_spaces OR has_punctuation)):
    → TEXT
```

Where:
- `has_spaces`: >50% of values contain whitespace
- `has_punctuation`: >30% of values contain `.,!?;:`

Text columns are stored in `self.text_features` and excluded from categorical encoding.

### TF-IDF Vectorizer Configuration

```python
TfidfVectorizer(
    max_features=5000,
    min_df=2,              # word must appear in >= 2 documents
    max_df=0.95,           # ignore words in > 95% of documents
    strip_accents='unicode',
    lowercase=True,
    ngram_range=(1, 2),    # unigrams + bigrams
    stop_words='english'
)
```

### Pipeline Integration

Each text column is vectorized independently. Results are concatenated via `scipy.sparse.hstack()`. Feature names follow the pattern `{column_name}_tfidf_{i}`.

The `ColumnTransformer` uses `sparse_threshold=0.3` to output sparse matrices when text features are present, maintaining memory efficiency.

The `TextVectorizer` class extends `BaseEstimator` and `TransformerMixin` for sklearn pipeline compatibility. Vectorizers are persisted with `joblib.dump()` alongside the model.

### Failure Handling

If vectorization fails for a column, a warning is logged and the column is skipped. No silent failures.

---

## Outlier Handling

Outlier treatment is model-family-specific and only applied when statistically necessary.

### Strategy by Model Family

| Model Family | Outlier Handling | Method | Rationale |
|---|---|---|---|
| LINEAR (Ridge, Lasso, Linear Regression) | Yes | IQR capping | Gradient descent is sensitive to extreme values; outliers distort coefficients |
| DISTANCE_BASED (KNN, SVM, K-Means, DBSCAN) | Yes | IQR capping | Euclidean distance dominated by extreme values; outliers skew cluster centers and decision boundaries |
| TREE_BASED (Random Forest, XGBoost, Decision Tree, LightGBM, Gradient Boosting) | No | — | Split decisions depend on value order, not magnitude; outliers may carry signal |

### IQR Capping Formula

```
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
x_capped = clip(x, lower_bound, upper_bound)
```

### OutlierCapper Implementation

Custom sklearn transformer (`BaseEstimator`, `TransformerMixin`):
- `fit()`: computes Q1, Q3, IQR bounds on training data
- `transform()`: clips values using `np.clip()` with stored bounds
- Placed as the first step in the numerical preprocessing pipeline (before imputation)
- Bounds are computed on training data only — no leakage into test set

### Detection Flow

1. Check model family membership
2. For LINEAR or DISTANCE_BASED: identify columns with significant outliers (using EDA results)
3. If outlier columns found: insert `OutlierCapper(iqr_multiplier=1.5)` into pipeline
4. Track `outliers_handled_count` and `outlier_columns` in metadata

---

## Typo Detection

### Algorithm

Uses Python's `difflib.SequenceMatcher` to compute pairwise similarity ratios between all unique categorical values in a column.

```python
similarity = SequenceMatcher(None, str(val1).lower(), str(val2).lower()).ratio()
```

### Threshold & Grouping

- **Similarity threshold**: >= 0.80 (80% match) flags a pair as potential typos
- **Direction**: the less frequent value is classified as the suspected typo; the more frequent value is the suggested correction
- **Significance filter**: only flagged if the typo affects >= 5 rows OR >= 1% of the dataset

### Performance Bounds

- Columns with > 100 unique values are skipped (avoids O(n^2) on high-cardinality columns)
- Complexity: O(n^2) where n = number of unique values (capped at 100)

### Detection Scope

Detects: misspellings (`Manager` vs `Manger`), case variants (`Male` vs `male`), trailing whitespace (`Active` vs `Active `), near-matches (`<50k` vs `<55k`).

Does not detect: semantically different valid categories, domain synonyms, values with < 80% similarity.

### Integration

Runs during EDA in `_analyze_categorical()`. Results appear in:
- `categorical_analysis[col]['potential_typos']` — array of typo objects
- EDA recommendations panel — warning strings

---

## Model Training & AutoML Engine

### Supported Models

**Classification** (9 models): Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, KNN, AdaBoost.

**Regression** (11 models): Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVR, KNN, AdaBoost.

**Clustering** (3 models): KMeans, DBSCAN, Agglomerative Clustering.

### Hyperparameter Optimization

- **Framework**: Optuna
- **Sampler**: TPESampler (Tree-structured Parzen Estimator)
- **Pruner**: MedianPruner (early stopping for poor trials)
- **Seed**: 42
- **Default trials per model**: 75 (configurable 10–200)
- **Cross-validation**: Stratified K-Fold for classification, K-Fold for regression (default 5 folds)

### Example Search Spaces

**Random Forest:**
```
n_estimators:      [50, 500]
max_depth:         [3, 30]
min_samples_split: [2, 20]
min_samples_leaf:  [1, 10]
max_features:      {sqrt, log2, None}
```

**XGBoost:**
```
n_estimators:      [50, 500]
learning_rate:     [0.001, 0.3]  (log scale)
max_depth:         [3, 10]
min_child_weight:  [1, 10]
subsample:         [0.5, 1.0]
colsample_bytree:  [0.5, 1.0]
gamma:             [0.0, 1.0]
```

### Training Flow

1. For each model: run Optuna optimization (N trials)
2. Each trial: fit model with sampled hyperparameters, compute CV score
3. After optimization: retrain with best params, compute train/CV/test scores
4. Calculate overfitting metrics and generalization score
5. Run stability checks on final model
6. Across all models: filter rejected models, select best by generalization score

---

## Model Selection & Generalization Scoring

### Formula

```
overfit_gap = |train_score - cv_score|
penalty = overfit_gap * penalty_factor
generalization_score = cv_score - penalty
```

### Penalty Tiers

| Overfit Gap | Penalty Factor | Action |
|---|---|---|
| > 0.20 | — | REJECTED (generalization_score set to -1.0) |
| 0.10 – 0.20 | 3.0 | High penalty |
| 0.05 – 0.10 | 2.0 | Medium penalty |
| < 0.05 | 2.0 | Low penalty |

### Selection

```python
valid_models = [m for m in models if not m.rejected]
best_model = max(valid_models, key=lambda m: m.generalization_score)
```

### Example

| Model | Train | CV | Gap | Penalty | Gen Score | Result |
|---|---|---|---|---|---|---|
| A | 0.95 | 0.87 | 0.08 | 0.16 | 0.71 | — |
| B | 0.89 | 0.86 | 0.03 | 0.06 | 0.80 | SELECTED |
| C | 0.98 | 0.75 | 0.23 | — | — | REJECTED |

Model B wins despite lower raw scores because it generalizes better.

### Configuration Defaults

```python
n_trials = 75
cv_folds = 5
test_size = 0.2
penalty_factor = 2.0
overfit_threshold_reject = 0.20
overfit_threshold_high = 0.10
```

---

## Model Stability Detection

Nine checks are performed on each model, both during Optuna trials and on the final trained model.

### The 9 Stability Checks

| # | Check | Condition | Applies To | Return Code |
|---|---|---|---|---|
| 1 | Convergence | `n_iter_ >= max_iter` | Iterative models (LogisticRegression, MLP, SGD, Lasso, Ridge, ElasticNet) | `convergence_failure` |
| 2 | NaN coefficients | `np.isnan(coef_).any()` | Linear models with `coef_` | `nan_coefficients` |
| 3 | Inf/large coefficients | `np.isinf(coef_)` or `abs(coef_) > 1e6` | Linear models with `coef_` | `inf_coefficients` / `large_coefficients` |
| 4 | NaN/Inf feature importances | `np.isnan/isinf(feature_importances_)` | Tree models | `nan_feature_importances` / `inf_feature_importances` |
| 5 | NaN/Inf predictions | `np.isnan/isinf(predictions)` | All | `nan_predictions` / `inf_predictions` |
| 6 | Single-class prediction | `len(unique(predictions)) == 1` | Classification | `single_class_prediction` |
| 7 | Zero variance predictions | `std(predictions) < 1e-10` | Regression | `zero_variance_predictions` |
| 8 | Extreme overfitting | `|train_score - cv_score| > 0.5` | All supervised | `extreme_overfitting` |
| 9 | Invalid CV score | `isnan(cv_score) or isinf(cv_score)` | All supervised | `invalid_cv_score` |

Additionally, prediction errors (exceptions during `model.predict()`) are caught and flagged as `prediction_error`.

### Optuna Integration

- `ConvergenceWarning` is captured via `warnings.catch_warnings(record=True)` during `model.fit()`
- Unstable trials return score `-9999`, causing Optuna to deprioritize that hyperparameter region
- Stable trials return their actual CV score

### Final Model Check

After training the best-params model, stability is checked again. Result is stored in `ModelResult.stability_status` (`"stable"` or `"unstable"`) and `ModelResult.stability_reason`. An unstable final model is not automatically rejected but is flagged for the user.

### Hardcoded Thresholds

- Large coefficient: `> 1e6`
- Extreme overfitting gap: `> 0.5`
- Zero variance: `std < 1e-10`

---

## GPU Configuration

### Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `XGB_GPU_MEMORY_FRACTION` | XGBoost GPU memory limit | `max_gpu_usage` (0.8) |
| `TF_FORCE_GPU_ALLOW_GROWTH` | TensorFlow dynamic allocation | `true` |
| `TF_GPU_MEMORY_FRACTION` | TensorFlow GPU limit | `max_gpu_usage` |
| `PYTORCH_CUDA_ALLOC_CONF` | PyTorch CUDA config | `max_split_size_mb:512` |

### Parameter Scaling

When `max_gpu_usage < 1.0`:

**XGBoost:**
```python
tree_method = 'hist'                          # memory-efficient histogram method
max_bin = int(256 * max_gpu_usage)            # scale bin count with allocation
```

**LightGBM:**
```python
max_bin = int(255 * max_gpu_usage)
min_data_in_bin = max(3, int(3 / max_gpu_usage))
```

### Memory Monitoring

Uses `psutil.Process().memory_percent()`. If usage exceeds 80%, `gc.collect()` is triggered automatically. Memory is checked before each model and cleaned up between models.

### GPU Allocation Slider

Frontend range: 20%–100%, step 5%, default 80%. Passed as `config.max_gpu_usage` (float 0.0–1.0) in the training request.

### Schema

```python
max_gpu_usage: float = Field(default=0.8, ge=0.0, le=1.0)
```

---

## API Reference

Base URL: `http://localhost:8000/api/automl`

### POST /train

Train models with Optuna optimization. Returns the best model by generalization score plus all per-model results.

**Request body:**
```json
{
  "dataset_id": "string (UUID)",
  "target_column": "string",
  "task_type": "classification | regression | clustering",
  "name": "string",
  "description": "string (optional)",
  "config": {
    "n_trials": 75,
    "cv_folds": 5,
    "test_size": 0.2,
    "penalty_factor": 2.0,
    "overfit_threshold_reject": 0.20,
    "overfit_threshold_high": 0.10,
    "max_gpu_usage": 0.8,
    "models_to_train": ["random_forest", "xgboost"]
  },
  "preprocessing_config": {
    "use_adaptive": true,
    "use_eda_insights": true,
    "force_drop_columns": [],
    "force_keep_columns": [],
    "imputation_overrides": {},
    "encoding_overrides": {},
    "disable_id_detection": false,
    "missing_threshold_override": null,
    "test_size": 0.2,
    "random_state": 42
  }
}
```

**Response (201):** Full training result including `best_model_name`, `best_generalization_score`, `best_cv_score`, `best_test_score`, `best_overfit_gap`, `best_params`, `all_models[]` (each with train/cv/test scores, overfit gap, penalty, generalization score, detailed metrics, feature importance, stability status), `plot_paths`, `model_path`, `preprocessing_path`, `training_duration`.

**Errors:** 400 (invalid dataset ID, missing target column), 404 (dataset not found), 500 (all models rejected).

### GET /models

List all trained models. Optional query param `dataset_id` to filter.

**Response (200):** Array of model summaries.

### GET /models/{model_id}

Full details for a single model. Same schema as the training response.

**Errors:** 400 (invalid ID), 404 (not found).

### DELETE /models/{model_id}

Deletes model, preprocessing pipeline, and plot files.

**Response:** 204 No Content.

**Errors:** 400 (invalid ID), 404 (not found).

### POST /api/datasets/{dataset_id}/preprocess

Run preprocessing on a dataset.

**Request body:**
```json
{
  "target_column": "string | null",
  "model_type": "string | null",
  "test_size": 0.2,
  "custom_config": {
    "drop_columns": [],
    "imputation_override": {}
  }
}
```

**Response (200):** Preprocessing summary with original/processed shape, changes dict, quality scores, step descriptions, removed/encoded/scaled column lists.

### Performance Characteristics

- Training time: ~2–5 min per model (data-size and n_trials dependent)
- Full 9-model run: ~20–45 min with defaults
- Memory: ~2–4 GB during training
- Concurrency: single training session
