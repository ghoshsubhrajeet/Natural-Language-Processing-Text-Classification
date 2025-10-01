## Text Classification — TF‑IDF + Logistic Regression

A compact text classification project for an Applied NLP assignment. You will build TF‑IDF features, add simple metadata features, and train Logistic Regression baselines. The implementation lives in a Jupyter notebook and is validated via automated tests.

### Objectives
- Obtain a TF‑IDF text representation and encode additional features
- Create, train, and run Logistic Regression models for text classification

### Repository structure
```text
text-classification-spectrasubhrajeet/
  ├─ data/
  │  ├─ train.tsv
  │  └─ test.tsv
  ├─ test_utils/                 # Golden artifacts used by tests
  │  ├─ vectorizer.pkl
  │  ├─ train_x_vectorized.pkl
  │  ├─ test_x_vectorized.pkl
  │  ├─ model.pkl
  │  ├─ balanced_model.pkl
  │  ├─ prediction.pkl
  │  ├─ train.tsv
  │  └─ test.tsv
  ├─ textclassification.ipynb    # Implement required functions here
  ├─ nbimport.py                 # Helper to import notebook as a module
  ├─ test.py                     # PyTest test suite
  ├─ conftest.py                 # Scoring summary for tests
  ├─ requirements.txt
  └─ README.md
```

## Setup
First, ensure you have Python 3.10+ installed.

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
pip install -r requirements.txt
```

If you prefer conda/mamba, create and activate an environment with Python 3.10+ and then install `requirements.txt`.

## Data
Training and test splits are provided under `data/`. The tests also reference fixed copies in `test_utils/` to validate your outputs against known-good artifacts.

## What to implement (in `textclassification.ipynb`)
Implement the following functions exactly with the specified signatures so that they can be imported and tested:

- `create_tfidfvectorizer()`: return a configured `sklearn.feature_extraction.text.TfidfVectorizer` matching parameters expected by tests.
- `run_vectorizer(vectorizer, train_series, test_series)`: fit on training text and transform both train/test, returning `(train_x, test_x)` sparse matrices.
- `create_model()`: return a baseline `sklearn.linear_model.LogisticRegression` with expected parameters.
- `run_model(model, train_x, train_y, test_x)`: fit the model and return predictions for `test_x`.
- `create_balanced_model()`: return a `LogisticRegression` configured for class imbalance (e.g., `class_weight="balanced"` as expected by tests).
- `create_column_transformer()`: return a `sklearn.compose.ColumnTransformer` combining:
  - TF‑IDF over the text field (column name: `"tweet"`), and
  - One‑hot encoding over the additional metadata feature (column name: `"sentiment"`).

The tests import these functions from the notebook using `nbimport.py`, so ensure the function names and signatures match exactly and are executed in code cells.

## Running the tests
Activate your environment, then run:

```bash
pytest -q
```

On success, you will see individual test scores and a Total Score reported by the custom summary in `conftest.py`.

## Tips
- Keep the exact parameterization consistent with the expected artifacts in `test_utils/` (e.g., tokenizer, n‑gram range, min_df/max_df, regularization, solver).
- Ensure `run_vectorizer` returns matrices whose values and shapes match the provided gold matrices.
- For deterministic behavior, avoid randomness or explicitly set seeds where applicable.

## Grading breakdown
- `test_create_tfidfvectorizer` — 16%
- `test_run_vectorizer` — 16%
- `test_create_model` — 16%
- `test_run_model` — 16%
- `test_create_balanced_model` — 16%
- `test_create_column_transformer` — 20%

## Environment
Key dependencies are pinned in `requirements.txt`:

```text
ipython, jupyter, notebook, pytest, pandas, numpy, scikit-learn
```

## License
If you plan to publish this repository publicly, add a `LICENSE` file (e.g., MIT) appropriate for your course policy. If this is coursework, follow your institution’s academic integrity rules.
