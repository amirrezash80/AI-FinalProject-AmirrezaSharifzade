# گزارش نهایی فاز دوم — آموزش، ارزیابی، بهبود و دمو (مسیر ۲: یادگیری ماشین کلاسیک)

**هدف این فاز** این بود که بعد از آماده‌سازی داده‌ها در فاز اول، چند مدل کلاسیک را به‌صورت منصفانه مقایسه کنم، سپس با Hyperparameter Tuning کیفیت را بهبود بدهم و در نهایت یک دمو قابل ارائه بسازم.

---

## 1) خلاصه‌ی مسیر اجرا (End-to-End)

1. **مقایسه‌ی مدل‌ها با تنظیمات پیش‌فرض (Baseline Comparison)**  
   مدل‌های اصلی: Logistic Regression، KNN، SVM (RBF)، Random Forest  
   (XGBoost در این اجرا وارد نشد.)

2. **Hyperparameter Tuning** با `RandomizedSearchCV` روی Train (با CV=3) و انتخاب بهترین مدل بر اساس **F1**.

3. **Threshold Tuning** روی Validation برای بیشینه کردن F1 و بررسی Trade-off ها (Precision/Recall) روی Test.  
   برای قابل اعتمادتر شدن احتمال‌ها، از **Calibration با روش sigmoid** هم استفاده کردم.

4. **دمو**: یک دمو Streamlit + یک ابزار CLI برای پیش‌بینی روی فایل CSV.

---

## 2) دیتاست و تعریف مسئله

- مسئله: **طبقه‌بندی دودویی** (Binary Classification)  
- ستون هدف: `Class`  
- نگاشت برچسب‌ها: `numeric_binary(1->0, 2->1)`  
- Seed ثابت برای تکرارپذیری: `42`  
- نکته‌ی پیش‌پردازش: `drop_duration=True` (برای جلوگیری از leakage در صورت وجود ویژگی‌های مشکل‌دار)

> در این پروژه من فرض کردم بخشی از ویژگی‌ها نام‌گذاری ناشناس دارند (مثل `V1..`) و بنابراین تحلیل را بر اساس رفتار مدل‌ها و معیارها انجام دادم، نه تفسیر معناییِ نام ویژگی‌ها.

---

## 3) نتایج Baseline (مدل پایه)

به عنوان baseline، **Logistic Regression** را اجرا کردم:

- **Validation**:  
  Accuracy=0.842 ، F1=0.542 ، ROC-AUC=0.904
- **Test**:  
  Accuracy=0.838 ، F1=0.532 ، ROC-AUC=0.902

---

## 4) مقایسه‌ی مدل‌ها (بدون Tuning)

در مقایسه‌ی اولیه (با تنظیمات پیش‌فرض):

- **KNN** دقت (Accuracy) بالایی داشت، اما Recall پایین → برای کلاس مثبت، نمونه‌های زیادی را از دست می‌داد.  
- **SVM-RBF** ROC-AUC خوبی داشت و نسبت به KNN Recall بهتر بود.  
- **Random Forest** در این مرحله Precision بالا ولی Recall پایین داشت (threshold=0.5 باعث محافظه‌کاری می‌شد).

(جدول و گزارش کامل در فایل‌های خروجی ذخیره شده است.)

---

## 5) Hyperparameter Tuning و انتخاب بهترین مدل

برای هر مدل، `RandomizedSearchCV` با `cv=3` و `n_iter=30` اجرا شد و معیار انتخاب **F1** بود.  
بهترین مدل نهایی:

- **Best Model**: `RandomForest`
- **Best CV F1**: `0.591`
- **Best Params**:
```json
{
  "model__max_depth": 20,
  "model__max_features": "sqrt",
  "model__min_samples_leaf": 3,
  "model__min_samples_split": 6,
  "model__n_estimators": 846
}
```

### عملکرد بهترین مدل (بعد از Tuning)

**Validation (thr=0.5):**
- Accuracy=0.887
- Precision=0.510
- Recall=0.739
- F1=0.604
- ROC-AUC=0.925

**Test (thr=0.5):**
- Accuracy=0.885
- Precision=0.507
- Recall=0.714
- F1=0.593
- ROC-AUC=0.921

---

## 6) Threshold Tuning + Calibration (sigmoid)

روی Validation یک sweep انجام دادم و سپس threshold بهینه را برای بیشینه کردن F1 انتخاب کردم:

- **Calibration**: `sigmoid`
- **Best Threshold (on Valid)**: 0.22
- (Valid) Precision=0.498, Recall=0.770, F1=0.605

### مقایسه روی Test: threshold=0.5 vs threshold بهینه

**Test @0.5**
- Accuracy=0.905
- Precision=0.635
- Recall=0.443
- F1=0.522
- ROC-AUC=0.921

**Test @best_threshold (0.22)**
- Accuracy=0.880
- Precision=0.492
- Recall=0.743
- F1=0.592
- ROC-AUC=0.921

**Trade-off:** threshold=0.5 → Precision/Accuracy بهتر، threshold بهینه → Recall/F1 بهتر.

---

## 7) نتیجه‌ی نهایی (Final Decision)

در ارائه، من هر دو خروجی را گزارش می‌کنم:

- **`pred@0.5`** برای سناریوهای Precision-محور  
- **`pred@best_threshold`** برای سناریوهای Recall/F1-محور

---

## 8) نحوه اجرای فاز دوم (Reproducibility)

### (الف) مقایسه اولیه مدل‌ها
```bash
python -m src.training.train_compare_models --out_dir results --save_models
```

### (ب) Hyperparameter Tuning و ذخیره بهترین مدل
```bash
python -m src.training.tune_models --out_dir results --n_iter 30 --cv 3 --scoring f1
```

### (ج) Threshold tuning (و calibration)
```bash
python -m src.training.threshold_tune_best --out_dir results --metric f1 --calibrate sigmoid
```

### (د) دمو Streamlit
```bash
pip install streamlit
streamlit run app/streamlit_app.py
```

### (هـ) inference روی CSV با CLI
```bash
python -m src.inference.predict_csv --input_csv data/raw/bank_marketing.csv --output_csv results/predictions_raw.csv
```

---

## 9) نکات و محدودیت‌ها

برای بهبودهای بیشتر می‌توان:
- Balanced sampling / SMOTE
- cost-sensitive learning
- Boosting (در صورت امکان نصب XGBoost)
- تحلیل Feature Importance کامل‌تر و انتخاب ویژگی

---

**پایان فاز دوم**
