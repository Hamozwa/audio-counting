"""
process_countix_av.py

Preprocess CountixAV or ExtremeCountixAV dataset to spectrogram .npy files for use in testing.

"""
# IMPORTS AND HELPER FUNCTIONS

import os
import csv
import torchaudio
from data_converter import DataConverter

def _get_six_digit(num: int) -> str:
    return str(num).zfill(6)

def to_int(x: str) -> int:
    return int(float(x))


# CONFIGURATION

# mode = "extreme"
mode = "countix"

hist_eq = None

count_limit = 8
resample_rate = 16000


converter = DataConverter()
ids = []
counts = []

# MODE 1: CountixAV

if mode == "countix":

    split = "test"

    folder = f"/datasets/Kinetics700-2020/{split}"
    # csv_file_path = f"/datasets/Kinetics700-2020/annotations/countix/countix_{split}.csv"
    csv_file_path = "/users/hani/AudioCounting/CountixAV_test.csv"
    output_folder = f"/scratch/local/ssd/hani/countix-av-nohist-spec/{split}/"

    # classes = [
    #     'battle rope training', 'bouncing ball (not juggling)', 'bouncing on trampoline',
    #     'clapping', 'gymnastics tumbling', 'juggling soccer ball', 'jumping jacks',
    #     'mountain climber (exercise)', 'planing wood', 'playing ping pong', 'playing tennis',
    #     'running on treadmill', 'sawing wood', 'skipping rope', 'slicing onion',
    #     'swimming backstroke', 'swimming breast stroke', 'swimming butterfly stroke',
    #     'swimming front crawl', 'swimming with dolphins', 'swimming with sharks',
    #     'tapping pen', 'using a wrench', 'using a sledge hammer'
    # ]

    # Load CSV + build file list
    with open(csv_file_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            count = int(float(row["count"]))
            if count <= count_limit:
                
                vid = (
                    f"Kinetics700-2020-test/"
                    f"{row['video_id']}_"
                    f"{_get_six_digit(int(row['kinetics_start']))}_"
                    f"{_get_six_digit(int(row['kinetics_end']))}.mp4"
                )

                ids.append(os.path.join(folder, vid))
                counts.append(count)



# MODE 2: ExtremeCountixAV

elif mode == "extreme":

    folder = "/scratch/local/ssd/hani/ExtremeCountixAV/"
    csv_file_path = os.path.join(folder, "ExtremeLabels.csv")
    audio_root = os.path.join(folder, "Audio")
    output_folder = "/scratch/local/ssd/hani/extreme-spec/"

    with open(csv_file_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            count = to_int(row["number_of_repetitions"])
            if count > count_limit:
                print(f"Skipping count {count} for video_id {row['youtube_id']}")
                continue

            video_id = row["youtube_id"]
            action_class = row["action_class"].strip() if row["action_class"] else ""

            # # Case 1 — direct subfolder
            # if action_class:
            #     audio_path = os.path.join(audio_root, action_class, video_id + ".wav")

            #     if not os.path.exists(audio_path):
            #         print(f"Warning: missing {audio_path}")
            #         continue

            #     ids.append(audio_path)

            # # Case 2 — search subfolders
            # else:
            #     found = False
            #     for subfolder in os.listdir(audio_root):
            #         candidate = os.path.join(audio_root, subfolder, video_id + ".wav")
            #         if os.path.exists(candidate):
            #             ids.append(candidate)
            #             found = True
            #             break

            #     if not found:
            #         print(f"Warning: could not find audio for video_id {video_id}")
            #         continue

            ids.append(os.path.join(audio_root, "ALL", video_id + ".wav"))



            counts.append(count)


# PROCESSING LOOP

for i, (file_path, count) in enumerate(zip(ids, counts)):

    if not os.path.exists(file_path):
        print(f"Warning: missing file {file_path}")
        continue

    # torchaudio.load works on wav and mp4
    y, sample_rate = torchaudio.load(file_path, normalize=True)
    y = torchaudio.transforms.Resample(sample_rate, resample_rate)(y)

    out_path = os.path.join(output_folder, f"{i}_{count}.npy")

    converter.create_spectrogram_npy(
        y, resample_rate,
        out_npy_path=out_path,
        hist_eq=hist_eq,
        add_noise=False
    )

    if i % 100 == 0:
        print(f"Processed {i} / {len(ids)}")
