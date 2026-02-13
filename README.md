# AI Final Project — مسیر ۲ (یادگیری ماشین صنعتی / Classic ML)

من توی این پروژه یک مسئله‌ی **طبقه‌بندی دودویی (Binary Classification)** روی یک دیتاست واقعیِ جدولی (Tabular) رو انتخاب کردم و هدفم اینه که کل مسیر صنعتیِ یک پروژه‌ی ML کلاسیک رو **مرحله‌به‌مرحله و قابل بازتولید** پیاده‌سازی کنم.

در فاز اول تمرکز روی **تحلیل مسئله، EDA، آماده‌سازی داده، Feature Engineering و یک مدل Baseline** هست.  
در فاز دوم، مدل‌های کلاسیک مختلف رو پیاده‌سازی و مقایسه می‌کنم (SVM / RandomForest / XGBoost / KNN / Logistic Regression) و ارزیابی کامل + بهبود + دمو ارائه می‌دم.

---

## 1) مسئله چیست؟

هدف: پیش‌بینی کلاس هدف (Label) براساس ویژگی‌های موجود در داده.  
این یک سناریوی کاملاً رایج در صنعت هست: داده‌های جدولی با ترکیب ویژگی‌های عددی و (گاهی) دسته‌ای، مقدارهای Missing/Unknown، و احتمالاً عدم‌توازن کلاس.

---

## 2) دیتاست

من یک دیتاست Tabular واقعی رو در مسیر `data/raw/bank_marketing.csv` قرار می‌دم.  
اسکریپت دانلود/لود هم داخل پروژه هست (`src/data/*`)؛ ولی اگر دانلود خودکار انجام نشه، فایل CSV رو دستی در همین مسیر می‌ذارم.

**نکته مهم:** برای اینکه پروژه در GitHub تمیز بمونه، پوشه‌های زیر در `.gitignore` قرار گرفتن و در Git کامیت نمی‌شن:
- `data/raw/`
- `data/processed/`
- `results/`
- `models/`

پس دیتاست و خروجی‌های حجیم را به ریپو push نمی‌کنم.

---

## 3) ساختار پروژه (ماژولار)

```
src/
  data/            # دانلود/لود داده
  features/        # پاکسازی + Feature Engineering + preprocess pipeline
  models/          # تعریف مدل‌ها و کانفیگ‌ها
  training/        # split + آماده‌سازی داده + آموزش
  evaluation/      # metrics + plots
  utils/           # ابزارها (seed, paths, ...)
notebooks/         # نوت‌بوک‌های فازها (با خروجی‌های کلیدی)
tests/             # تست‌های کوچک برای اجزای کلیدی
app/               # دمو (در فاز ۲)
```

---

## 4) راه‌اندازی محیط

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> روی macOS معمولاً `python` وجود ندارد و باید `python3` استفاده شود؛ داخل venv دستور `python` معمولاً کار می‌کند.

---

## 5) اجرای مراحل فاز اول

### 5.1) دانلود/لود دیتاست
```bash
python -m src.data.download_bank_marketing
```

اگر دانلود خودکار انجام نشد، فایل CSV را دستی قرار می‌دهم:
`data/raw/bank_marketing.csv`

### 5.2) اجرای EDA (نوت‌بوک)
نوت‌بوک فاز اول:
- `notebooks/phase1_full.ipynb`

در این نوت‌بوک:
- وضعیت داده و ستون‌ها بررسی می‌شود
- Missing/Unknown تحلیل می‌شود
- حداقل ۶ نمودار EDA رسم می‌شود
- نکات کلیدی برای Feature Engineering استخراج می‌شود
- Baseline Logistic Regression آموزش و ارزیابی می‌شود

**نکته تحویلی:** قبل از تحویل، نوت‌بوک را Run All می‌کنم و با خروجی (plots/tables) ذخیره می‌کنم تا خروجی بخش‌های کلیدی مشخص باشد.

### 5.3) آماده‌سازی داده و پیش‌پردازش (Split + Pipeline)
```bash
python -m src.training.prepare_data --save --drop_duration
```

خروجی‌ها (لوکال):
- `data/processed/preprocess.joblib`
- `data/processed/dataset_sparse.joblib`
- `data/processed/prepare_summary.json`

### 5.4) آموزش baseline (Logistic Regression)
```bash
python -m src.training.train_baseline --out_dir results
```

خروجی‌ها (لوکال):
- Confusion Matrix و ROC در `results/figures/`
- گزارش متریک‌ها در `results/tables/baseline_logreg_report.json`
- مدل ذخیره‌شده در `results/models/baseline_logreg.joblib`

---

## 6) معیارهای ارزیابی

برای ارزیابی علمی از موارد زیر استفاده می‌کنم:
- Accuracy
- Precision / Recall / F1-score
- ROC-AUC (در صورت وجود هر دو کلاس)
- Confusion Matrix
- ROC Curve

در فاز دوم علاوه بر این‌ها:
- Feature Importance (برای مدل‌های tree-based مثل RandomForest/XGBoost)
- جدول مقایسه‌ی مدل‌ها

---

## 7) قوانین GitHub و نحوه‌ی توسعه

### نام ریپو
`AI-FinalProject-<GroupName>`

### Branch ها
- فاز اول: `phase-1`
- فاز دوم: `phase-2`

هر فاز روی برنچ خودش توسعه داده می‌شود و بعد از تکمیل با Pull Request به `main` مرج می‌شود.

### Commit های معنادار
در هر فاز حداقل ۵ commit معنادار می‌زنم.  
نمونه پیام‌های قابل قبول:
- `Add Phase-1 EDA notebook with exploratory plots and findings`
- `Implement preprocessing pipeline with stratified split and feature engineering`
- `Add Logistic Regression baseline training with ROC/CM plots`

---

## 8) پلن فاز دوم (خلاصه)

در فاز دوم:
- پیاده‌سازی و مقایسه‌ی مدل‌ها:
  - Logistic Regression (baseline)
  - SVM
  - KNN
  - RandomForest
  - XGBoost
- Hyperparameter Tuning (Grid/Random Search)
- بهبود با Feature Engineering و انتخاب ویژگی
- گزارش کامل و دمو (Streamlit)

