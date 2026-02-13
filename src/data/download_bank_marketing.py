from __future__ import annotations
import argparse
from pathlib import Path

from sklearn.datasets import fetch_openml
from src.utils.paths import DATA_RAW

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(DATA_RAW / "bank_marketing.csv"))
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    try:
        # OpenML: Bank Marketing dataset (UCI)
        ds = fetch_openml(name="bank-marketing", version=1, as_frame=True)
        df = ds.frame
        # Normalize target column name (usually 'y': yes/no)
        if "class" in df.columns and "y" not in df.columns:
            df = df.rename(columns={"class": "y"})
        df.to_csv(out_path, index=False)
        print(f"Saved dataset to: {out_path} (rows={len(df)}, cols={df.shape[1]})")
    except Exception as e:
        print("Auto-download failed:", repr(e))
        print("\nFallback option:")
        print("1) دانلود دستی Bank Marketing CSV (UCI/OpenML) و قرار دادن در مسیر data/raw/bank_marketing.csv")
        print("2) سپس مرحله بعدی را اجرا کنید.")

if __name__ == "__main__":
    main()
