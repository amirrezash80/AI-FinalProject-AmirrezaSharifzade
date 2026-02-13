# AI Final Project — مسیر ۲: یادگیری ماشین صنعتی (Classic ML)

این پروژه مطابق الزامات «مسیر ۲» طراحی می‌شود: کار با دیتاست واقعی، Feature Engineering کامل، Split اصولی،
مقایسه‌ی مدل‌های کلاسیک (SVM / RandomForest / XGBoost / KNN / LogisticRegression) و ارزیابی علمی
(Confusion Matrix / ROC Curve / Feature Importance).  (طبق PDF درس)

## موضوع/دیتاست انتخابی
**Bank Marketing (Deposit Subscription Prediction)**  
یک دیتاست واقعی Tabular با ویژگی‌های عددی/دسته‌ای، عدم‌توازن برچسب و مقادیر Unknown.

چالش‌های واقعی که پوشش می‌دهیم:
- داده‌ی ترکیبی (categorical + numeric) → Encoding/Scaling
- Missing/Unknown values → پاکسازی و Imputation
- Class Imbalance → metrics مناسب + class_weight/threshold
- Leakage feature (duration) → تصمیم علمی برای حذف/مقایسه

## ساختار پروژه (ماژولار)
```
src/
  data/            # دانلود/لود دیتاست
  features/        # Feature Engineering + preprocess pipeline
  models/          # تعریف مدل‌ها و hyperparams
  training/        # train/validate pipelines
  evaluation/      # metrics + plots
  utils/           # ابزارهای عمومی (paths, seed)
notebooks/         # نوت‌بوک‌های فاز ۱/۲ (با خروجی‌های کلیدی)
app/               # دمو (Streamlit)
```

## قوانین GitHub (الزامی)
- Branch فاز ۱: `phase-1`
- Branch فاز ۲: `phase-2` + Pull Request برای Merge به `main`
- حداقل ۵ Commit معنادار در هر فاز

## اجرای سریع (بعد از مرحله‌ی بعدی)
```bash
python -m src.data.download_bank_marketing
```
# AI-FinalProject-AmirrezaSharifzade
