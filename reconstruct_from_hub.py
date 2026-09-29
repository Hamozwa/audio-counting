"""
Reorganise downloaded HF synthetic dataset repo to fit original layout used by tester scripts.
"""

import json
import os

from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm

REPO_ID = "Hamozwa/RepeatSynth"
OUT_DIR = "."

configs = {"rs": "RS", "rsn": "RSN", "rvn": "RVN"}
splits = {"train": "train", "validation": "val", "test": "test"}

for config_name, folder_name in configs.items():
    for hub_split, local_split in splits.items():
        try:
            ds = load_dataset(REPO_ID, config_name, split=hub_split)
        except Exception as e:
            print(f"skipping {folder_name}/{local_split}: {e}")
            continue

        split_dir = os.path.join(OUT_DIR, folder_name, "wav", local_split)
        os.makedirs(split_dir, exist_ok=True)

        meta = {}
        for ex in tqdm(ds, desc=f"{folder_name}/{local_split}"):
            sf.write(os.path.join(split_dir, f"{ex['id']}.wav"), ex["audio"]["array"], ex["audio"]["sampling_rate"])
            meta[ex["id"]] = {
                "original_source": ex["original_source"],
                "num_repetitions": ex["num_repetitions"],
                "events": ex["events"],
            }

        json.dump(meta, open(os.path.join(split_dir, "metadata.json"), "w"))
        print(f"{folder_name}/{local_split}: {len(meta)} examples")
