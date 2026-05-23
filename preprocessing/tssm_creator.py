import torch
import torchaudio
import torchaudio.functional as F
import torch.nn.functional as FT
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from data_converter import DataConverter
from tqdm import tqdm

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

if device.type == "cpu":
    print("Warning: Running on CPU")

wav_2_vec_bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav_2_vec_model = wav_2_vec_bundle.get_model().to(device)
converter = DataConverter()

def wav_2_vec(file, normalise=False, info_dict=None):
    waveform, sr = torchaudio.load(file)
    waveform = waveform.to(device)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if info_dict is not None:
        code = os.path.basename(file).replace(".wav", "")
        _, start_time = info_dict[code]
        start_idx = int(start_time * 16000)
        waveform = waveform[:, start_idx:]

    if normalise:
        mean = waveform.mean()
        std = waveform.std()
        waveform = (waveform - mean) / (std + 1e-8)
    
    with torch.no_grad():
        features, _ = wav_2_vec_model(waveform)
    return features

def create_tssm(file, histogram_equalisation=False, info_dict=None):
    features = wav_2_vec(file, info_dict=info_dict)
    x = features[0]

    x=FT.normalize(x, p =2, dim =-1)
    tssm = (x @ x.T).detach().cpu().numpy()

    if histogram_equalisation:
        tssm = converter.apply_histogram_equalisation(tssm)
    return tssm

def batch_create_tssm(input_folder, output_folder, csv_used=False, csv_path=None, start_edit=False):
    files = os.listdir(input_folder)
    info_dict = None
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
            #recreate files list based on csv file
            files = []
            for code in info_dict.keys():
                if os.path.exists(os.path.join(input_folder, code + ".wav")):
                    files.append(code + ".wav")
            
    no_of_files = len(files)
    for item in tqdm(files, desc=f"Processing {os.path.basename(input_folder.strip('/'))}", leave=False):
        file = os.path.join(input_folder, item)
        if file.endswith(".wav"):
            tssm = create_tssm(file, histogram_equalisation=False, info_dict=info_dict if start_edit else None)
            if csv_used:
                label, start_time = info_dict[item.replace(".wav", "")]
                save_path = os.path.join(output_folder, item.replace('.wav', f'_{label}.npy'))
            else:
                save_path = os.path.join(output_folder, item.replace('.wav', '.npy'))
            np.save(save_path, tssm)

# input_folder="/scratch/local/hdd/hani/bbc_clocks/audio/"
# output_folder="/scratch/local/hdd/hani/bbc_clocks/tssm/test/"

# csv_path = "/users/hani/AudioCounting/preprocessing/bbc_clocks/bbc_clocks.csv"
# csv_used = True
# start_edit = True

# batch_create_tssm(input_folder, output_folder, csv_used=csv_used, csv_path=csv_path, start_edit=start_edit)

# input_folder = "/scratch/local/hdd/hani/audioset_eval/audio_mono/"
# output_folder = "/scratch/local/ssd/hani/audioset_eval/tssm/test/"