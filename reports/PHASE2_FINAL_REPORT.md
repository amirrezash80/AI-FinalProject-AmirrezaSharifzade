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

## 4) مقایسه مدل‌ها (Baselineها و Phase-2)

در این فاز، ۴ مدل را روی مجموعه‌ی **Valid** و **Test** مقایسه کردم: Logistic Regression (Baseline)، KNN، Random Forest و SVM-RBF.
برای یک مقایسه‌ی سریع، جدول زیر را از خروجی `phase2_model_comparison.csv` گذاشتم (همه‌ی F1ها در این جدول با آستانه‌ی پیش‌فرض 0.5 گزارش شده‌اند):

| مدل | AUC (Valid) | F1 (Valid @0.5) | AUC (Test) | F1 (Test @0.5) |
|---|---:|---:|---:|---:|
| LogReg | 0.904 | 0.542 | 0.902 | 0.532 |
| SVM-RBF | 0.920 | 0.505 | 0.916 | 0.515 |
| RandomForest | 0.926 | 0.417 | 0.921 | 0.440 |
| KNN | 0.887 | 0.399 | 0.872 | 0.385 |


**برداشت من از جدول بالا:**  
- از نظر **AUC**، مدل‌های **RandomForest** و **SVM-RBF** تقریباً بهترین هستند (حدود 0.92).  
- از نظر **F1 با آستانه‌ی 0.5**، **LogReg** کمی بهتر از بقیه می‌افتد، ولی در عوض تعداد **False Positive**‌هایش زیادتر است (precision پایین‌تر).  
- چون داده نامتوازن است، عدد AUC به تنهایی کافی نیست و در عمل **انتخاب آستانه** خیلی روی F1/recall/precision اثر می‌گذارد (که در بخش 6 انجامش دادم).

### 4.2) نمودارهای مدل‌ها (Confusion Matrix / ROC / اهمیت ویژگی‌ها)

برای اینکه فقط به عددها اکتفا نکنم، خروجی‌های تصویری هر مدل را هم در پوشه‌ی `figures/` گذاشتم و اینجا لینکشان را قرار دادم.  
(این تصاویر دقیقاً همان‌هایی هستند که در خروجی کد تولید شده‌اند.)

#### Logistic Regression (Baseline)
- Confusion Matrix (Valid):  
![](../figures/baseline_logreg_cm_valid.png)

- Confusion Matrix (Test):  
![](../figures/baseline_logreg_cm_test.png)

- ROC (Valid):  
![](../figures/baseline_logreg_roc_valid.png)

- ROC (Test):  
![](../figures/baseline_logreg_roc_test.png)

#### KNN
- Confusion Matrix (Valid):  
![](../figures/KNN_cm_valid.png)

- Confusion Matrix (Test):  
![](../figures/KNN_cm_test.png)

- ROC (Valid):  
![](../figures/KNN_roc_valid.png)

- ROC (Test):  
![](../figures/KNN_roc_test.png)

#### Logistic Regression (Phase-2)
- Confusion Matrix (Valid):  
![](../figures/LogReg_cm_valid.png)

- Confusion Matrix (Test):  
![](../figures/LogReg_cm_test.png)

- ROC (Valid):  
![](../figures/LogReg_roc_valid.png)

- ROC (Test):  
![](../figures/LogReg_roc_test.png)

- Top-20 Coefficients:  
![](../figures/LogReg_coef_top20.png)

#### Random Forest (Phase-2)
- Confusion Matrix (Valid):  
![](../figures/RandomForest_cm_valid.png)

- Confusion Matrix (Test):  
![](../figures/RandomForest_cm_test.png)

- ROC (Valid):  
![](../figures/RandomForest_roc_valid.png)

- ROC (Test):  
![](../figures/RandomForest_roc_test.png)

- Top-20 Feature Importance:  
![](../figures/RandomForest_feature_importance_top20.png)

#### SVM-RBF
- Confusion Matrix (Valid):  
![](../figures/SVM-RBF_cm_valid.png)

- Confusion Matrix (Test):  
![](../figures/SVM-RBF_cm_test.png)

- ROC (Valid):  
![](../figures/SVM-RBF_roc_valid.png)

- ROC (Test):  
![](../figures/SVM-RBF_roc_test.png)

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

## 6) Threshold Tuning + Calibration (Final Best Pipeline)

چون دیتاست نامتوازن است، استفاده از آستانه‌ی ثابت 0.5 همیشه بهترین تصمیم نیست. برای همین، من برای **بهترین پایپ‌لاین نهایی** این کارها را انجام دادم:

1) **کالیبراسیون** احتمال‌ها با روش **Sigmoid** (روی Valid)  
2) **جست‌وجوی آستانه** روی Valid با معیار **F1** و انتخاب آستانه‌ی بهینه

نتیجه‌ی انتخاب آستانه روی **Valid**:
- آستانه‌ی بهینه روی Valid: **thr ≈ 0.22**
- (Valid) Precision: **0.498**
- (Valid) Recall: **0.770**
- (Valid) F1: **0.605**

ارزیابی روی **Test**:
- با thr=0.50 → Precision **0.635** / Recall **0.443** / F1 **0.522**
- با thr≈0.22 → Precision **0.492** / Recall **0.743** / F1 **0.592**

نکته‌ی مهم: **AUC** با تغییر آستانه تغییر نمی‌کند و برای Test حدود **0.921** است، اما با انتخاب آستانه‌ی درست می‌توانم trade-off بین precision و recall را مطابق نیاز پروژه تنظیم کنم.

### 6.1) نمودارهای کالیبراسیون، ROC/PR و انتخاب آستانه

- Calibration Curve (Valid) — بعد از کالیبراسیون Sigmoid:  
![](../figures/phase2_calibration_curve_valid.png)

- ROC (Valid) برای Best Pipeline:  
![](../figures/phase2_roc_valid_bestpipe.png)

- Precision–Recall (Valid) (AP≈0.61):  
![](../figures/phase2_pr_curve_valid.png)

- F1 vs Threshold (Valid) — برای پیدا کردن آستانه‌ی بهینه (حدود 0.22):  
![](../figures/phase2_f1_vs_threshold_valid.png)

- Confusion Matrix روی Test با آستانه‌های مختلف:  
  - thr=0.50  
![](../figures/phase2_cm_test_threshold_0.5.png)

  - thr=0.22 (بهینه روی Valid)  
![](../figures/phase2_cm_test_threshold_0.22.png)

  - thr=0.49 (نزدیک به 0.5 ولی با trade-off متفاوت)  
![](../figures/phase2_cm_test_threshold_0.49.png)

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
