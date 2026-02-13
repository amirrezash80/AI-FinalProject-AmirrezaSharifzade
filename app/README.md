# Demo (Streamlit)

اجرای دمو:
```bash
pip install streamlit
streamlit run app/streamlit_app.py
```

ورودی:
- یک فایل CSV که ستون‌های دیتاست را داشته باشد (وجود ستون هدف هم مشکلی ندارد؛ خودکار حذف می‌شود)

خروجی:
- جدول پیش‌بینی (probability + pred@0.5 و اگر وجود داشته باشد pred@best_threshold)
- امکان دانلود خروجی به صورت CSV
