# فاز دوم — آموزش، ارزیابی، بهبود و دمو (Classic ML)

در این فاز من چند مدل کلاسیک رو مقایسه کردم، بعد با Hyperparameter Tuning بهترین مدل رو انتخاب کردم، threshold رو تنظیم کردم و در نهایت یک دمو Streamlit و یک ابزار CLI برای inference ساختم.

---

## 0) نصب و اجرا

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## 1) مقایسه مدل‌ها (Baseline Comparison)

```bash
python -m src.training.train_compare_models --out_dir results --save_models
```

خروجی‌ها:
- `results/tables/phase2_model_comparison.csv`
- `results/tables/phase2_model_reports.json`
- نمودارها در `results/figures/`

---

## 2) Hyperparameter Tuning و انتخاب Best Model

```bash
python -m src.training.tune_models --out_dir results --n_iter 30 --cv 3 --scoring f1
```

خروجی‌ها:
- `results/tables/phase2_tuning_results.csv`
- `results/tables/phase2_best_model_report.json`
- `results/models/phase2_best_model.joblib`

---

## 3) Threshold Tuning + Calibration

```bash
python -m src.training.threshold_tune_best --out_dir results --metric f1 --calibrate sigmoid
```

خروجی‌ها:
- `results/tables/phase2_threshold_sweep_valid.csv`
- `results/tables/phase2_threshold_summary.json`
- نمودارها در `results/figures/`

---

## 4) دمو Streamlit

```bash
pip install streamlit
streamlit run app/streamlit_app.py
```

- دمو از `results/models/phase2_best_model.joblib` استفاده می‌کند.
- اگر `results/tables/phase2_threshold_summary.json` موجود باشد، `pred@best_threshold` هم نمایش داده می‌شود.

---

## 5) CLI برای inference روی CSV

```bash
python -m src.inference.predict_csv --input_csv data/raw/bank_marketing.csv --output_csv results/predictions_raw.csv
```

---

---

## 6) تصاویر کلیدی (برای ارائه)

> مسیر پیشنهادی: فایل‌ها داخل `results/figures/` ذخیره می‌شوند و این README همان مسیر را reference می‌کند.

**Baseline LogReg — Confusion Matrix (Valid)**

![](results/figures/baseline_logreg_cm_valid.png)

**Baseline LogReg — Confusion Matrix (Test)**

![](results/figures/baseline_logreg_cm_test.png)

**Baseline LogReg — ROC (Valid)**

![](results/figures/baseline_logreg_roc_valid.png)

**Baseline LogReg — ROC (Test)**

![](results/figures/baseline_logreg_roc_test.png)

**KNN — Confusion Matrix (Valid)**

![](results/figures/KNN_cm_valid.png)

**KNN — Confusion Matrix (Test)**

![](results/figures/KNN_cm_test.png)

**KNN — ROC (Valid)**

![](results/figures/KNN_roc_valid.png)

**KNN — ROC (Test)**

![](results/figures/KNN_roc_test.png)

**LogReg (Tuned) — Confusion Matrix (Valid)**

![](results/figures/LogReg_cm_valid.png)

**LogReg (Tuned) — Confusion Matrix (Test)**

![](results/figures/LogReg_cm_test.png)

**LogReg (Tuned) — ROC (Valid)**

![](results/figures/LogReg_roc_valid.png)

**LogReg (Tuned) — ROC (Test)**

![](results/figures/LogReg_roc_test.png)

**LogReg — Top-20 |coef|**

![](results/figures/LogReg_coef_top20.png)

**RandomForest — Confusion Matrix (Valid)**

![](results/figures/RandomForest_cm_valid.png)

**RandomForest — Confusion Matrix (Test)**

![](results/figures/RandomForest_cm_test.png)

**RandomForest — ROC (Valid)**

![](results/figures/RandomForest_roc_valid.png)

**RandomForest — ROC (Test)**

![](results/figures/RandomForest_roc_test.png)

**RandomForest — Top-20 Feature Importance**

![](results/figures/RandomForest_feature_importance_top20.png)

## نتیجه نهایی (خلاصه)

- Best model (بعد از tuning): **RandomForest**
- Best CV F1: **0.591**
- Test ROC-AUC (تقریباً): **0.921**
- Best threshold (on valid): **0.22**

گزارش کامل:
- `reports/PHASE2_FINAL_REPORT.md`

Notebook ارائه:
- `notebooks/phase2_final_demo.ipynb`
