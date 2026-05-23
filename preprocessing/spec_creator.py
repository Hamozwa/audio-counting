import os
import torchaudio
from scipy import signal
import numpy as np
import csv

from data_converter import DataConverter

#create spectrogram version of datasets
def batch_create_spectrograms(input_dir, output_dir, csv_used=False, csv_path=None, start_edit=False):
    files = os.listdir(input_dir)
    if csv_used:
        #extract lines from csv file
        info_dict = {}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row["location"].replace(".wav", "")
                label = int(row["repetitions"])
                if start_edit:
                    start_time = float(row["start_time"])
                    info_dict[code] = [label, start_time]
                else:
                    info_dict[code] = [label, 0.0]

        files = []
        for code in info_dict.keys():
            if os.path.exists(os.path.join(input_dir, code + ".wav")):
                files.append(code + ".wav")

    converter = DataConverter()

    no_of_files = len(files)
    step_size = max(1, no_of_files // 500)
    current_file = 1
    for filename in files:
        if not filename.endswith(".wav"):
            continue
        y, sr = torchaudio.load(os.path.expanduser(os.path.join(input_dir, filename)))
        y = y.numpy()
        
        if csv_used:
            #get label + start time from info dict
            label, start_time = info_dict[filename.replace(".wav", "")]
            if start_time > 0:
                start_idx = int(start_time * sr)
                y = y[:, start_idx:]

        waveform = y.mean(axis=0)
        _, _, spectrogram = signal.spectrogram(waveform, sr, nperseg=512, noverlap=256)
        spectrogram = np.log(spectrogram + 1e-7)

        spectrogram = converter.apply_histogram_equalisation(spectrogram, method="global")

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        if spectrogram.ndim == 3:
            spectrogram = spectrogram.mean(axis=0)

        if csv_used:
            filename = f"{filename.replace('.wav', '')}_{label}.npy"
        else:
            filename = f"{filename.replace('.wav', '')}.npy"

        output_path = os.path.join(output_dir, filename)
        np.save(output_path, spectrogram)

        if current_file % step_size == 0:
            print(f"Processed {current_file}/{no_of_files} files.")
        current_file += 1

# input_dir = "/scratch/local/hdd/hani/audioset_eval/audio_mono/"
# output_dir = "/scratch/local/ssd/hani/audioset_eval/spec/test/"

# csv_path = "/users/hani/AudioCounting/preprocessing/audio_strong/rep_labels.csv"
# csv_used = True
# start_edit = False

# input_dir = "/scratch/local/ssd/hani/RS/wav/test/"
# output_dir = "/scratch/local/ssd/hani/RS/spec/test/"
# batch_create_spectrograms(input_dir, output_dir)

# input_dir = "/scratch/local/ssd/hani/RSN/wav/test/"
# output_dir = "/scratch/local/ssd/hani/RSN/spec/test/"
# batch_create_spectrograms(input_dir, output_dir)

# input_dir = "/scratch/local/ssd/hani/RVN/wav/test/"
# output_dir = "/scratch/local/ssd/hani/RVN/spec/test/"
# batch_create_spectrograms(input_dir, output_dir)

input_dir = "/scratch/local/hdd/hani/heartbeats/wav/"
output_dir = "/scratch/local/hdd/hani/heartbeats/spec/"
batch_create_spectrograms(input_dir, output_dir)