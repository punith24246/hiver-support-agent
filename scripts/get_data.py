"""Fetch twcs.csv into data/. Works locally and in Colab.

    python -m scripts.get_data

Needs Kaggle credentials (kagglehub will prompt, or set KAGGLE_USERNAME /
KAGGLE_KEY). If you already have the file, just drop it at data/twcs.csv and
skip this.
"""

import os
import shutil

DEST = "data/twcs.csv"


def main():
    if os.path.exists(DEST):
        print(f"{DEST} already present ({os.path.getsize(DEST)/1e6:.0f} MB)")
        return

    import kagglehub

    path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
    src = None
    for root, _dirs, files in os.walk(path):
        for fn in files:
            if fn.lower() == "twcs.csv":
                src = os.path.join(root, fn)
    if src is None:
        raise SystemExit(f"twcs.csv not found under {path}")

    os.makedirs("data", exist_ok=True)
    shutil.copy(src, DEST)
    print(f"-> {DEST} ({os.path.getsize(DEST)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
