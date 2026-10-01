# Bank Marketing - Term Deposit Subscription Prediction (MLOps project)

Predicts whether a client will subscribe to a term deposit (`y = yes`) from the UCI
Bank Marketing dataset (`bank-additional-full.csv`, semicolon-separated).

Cleaning (shared in `src/data_utils.py`): drop duplicate rows, treat `"unknown"` as missing in
six categorical columns, and drop `duration` (known only after the call, so it leaks the outcome)
plus the raw `y` column.

Run everything from the project root:

```bash
pip install -r requirements.txt
python pipelines/run_lab3_baseline.py   # preprocess -> train 8 models -> evaluate best
python pipelines/run_lab4_tracking.py   # MLflow experiment tracking + reproducibility check
python pipelines/run_lab5_pipeline.py   # schema validation -> sklearn pipeline -> output validation
python pipelines/run_lab6_registry.py   # register model -> champion/challenger -> deployment report
```

Labs 4 and 6 need `mlflow`; Lab 5 needs `pandera`. Run Lab 5 before Lab 6 (Lab 6 logs `models/preprocessor.pkl`).
The exploratory notebook is in `notebooks/`; production logic lives in `src/` and `pipelines/`.
