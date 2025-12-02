"""

split_audio.py

Script to split FSD50K audio files into train/val directories

"""

import csv
import os
import shutil


CSV_PATH = "/scratch/local/ssd/hani/FSD50K/FSD50K.ground_truth/dev.csv"
SRC_DIR = "/scratch/local/ssd/hani/FSD50K/FSD50K.dev_audio/"
OUT_DIRS = {"train":"/scratch/local/ssd/hani/FSD50K/train", "val":"/scratch/local/ssd/hani/FSD50K/val"}

# Read CSV
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        fname = row["fname"] + ".wav"
        split = row["split"]

        src = os.path.join(SRC_DIR, fname)
        dst = os.path.join(OUT_DIRS[split], fname)

        if not os.path.exists(src):
            print(f"Missing file: {src}")
            continue

        shutil.move(src, dst)
        print(f"Moved {fname} → {split}/")
