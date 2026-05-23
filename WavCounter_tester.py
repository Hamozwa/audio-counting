import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import datasets
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from time import time
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import json
import csv
from scipy.signal import find_peaks

import os

np.random.seed(0)
torch.manual_seed(0)

finetune = True #set to true to test finetuned w2v model, false to test transformer variant trained on w2v features

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if not finetune:

    wav_2_vec_bundle = torchaudio.pipelines.WAV2VEC2_BASE
    wav_2_vec_model = wav_2_vec_bundle.get_model()

    wav_2_vec_model = wav_2_vec_model.to(device)

    def standardize_audio(waveform):
        mu = waveform.mean()
        sigma = waveform.std()
        return (waveform - mu) / (sigma + 1e-7)

    def wav_2_vec(file, normalise=False, pitch_shift=False, volume_shift=False):
        waveform, sr = torchaudio.load(file)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if volume_shift:
            waveform /= waveform.abs().max() + 1e-7
        if pitch_shift:
            waveform = torchaudio.functional.pitch_shift(waveform, sr, n_steps=36)
        length = waveform.shape[1]/sr
        waveform = waveform.to(device)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if normalise:
            waveform = standardize_audio(waveform)
        with torch.no_grad():
            features, _ = wav_2_vec_model(waveform)
        return features, length

class AudioWithJSON(torch.utils.data.Dataset):
    def __init__(self, input_dir, normalise=True ,use_w2v=False):
        self.input_dir = input_dir
        self.use_w2v = use_w2v
        self.normalise = normalise

        with open(os.path.join(input_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
    
        self.file_names = sorted(list(self.metadata.keys()))

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        wav_path = os.path.join(self.input_dir, f'{file_name}.wav')

        if self.use_w2v:
            features, audio_length = wav_2_vec(wav_path, normalise=self.normalise)
            feat_length = features.shape[1]
        else:
            #use wav directly
            features, sr = torchaudio.load(wav_path)
            if sr != 16000:
                features = torchaudio.functional.resample(features, sr, 16000)
            audio_length = features.shape[1] / 16000
            feat_length = 499

            if self.normalise:
                features = (features - features.mean()) / (features.std() + 1e-7)

        item = self.metadata[file_name]
        label = item['num_repetitions']
        starts = []
        ends = []
        for event in item['events']:
            starts.append(event['start'])
            ends.append(event['end'])

        target = torch.zeros(feat_length)
        t = torch.arange(feat_length)
        sigma = 2.0

        #GAUSSIANS AT EVENT CENTERS
        # event_points = (np.array(starts) + np.array(ends)) / 2

        # #convert to time in wav_2_vec features
        # event_points = [int(event * feat_length / audio_length) for event in event_points]
        
        # for s in event_points:
        #     # Add gaussian for each repetition
        #     bump = torch.exp(-(t - s)**2 / (2 * sigma**2))
        #     target = torch.max(target, bump)

        #SCALE GAUSSIAN TO FIT EVENT DURATION
        for start, end in zip(starts, ends):
            s_idx = start * feat_length / audio_length
            e_idx = end * feat_length / audio_length
            center = (s_idx + e_idx) / 2
            
            sigma = max((e_idx - s_idx) / 4.0, 2.0) 
            
            bump = torch.exp(-(t - center)**2 / (2 * sigma**2))
            target = torch.max(target, bump)
            
            #print(f'Event from {start:.2f}s to {end:.2f}s, sidx {s_idx:.2f}, eidx {e_idx:.2f}, center at {center:.2f} with sigma {sigma:.2f}')
            
        return features[0], label, target


class transformer_blocks(nn.Module):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_features, 
            nhead=8,
            batch_first=True
        )
        self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.layer1 = nn.Linear(in_features, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, out_features)


    def forward(self, x):
        x = self.transformer_blocks(x)
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))
        return x

class W2V_mlp(nn.Module):
    def __init__(self, out_features=1, use_sigmoid=True, w2v_version='base'):
        super().__init__()
        if w2v_version == 'base':
            self.wav_2_vec = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
            input_dim = 768
        elif w2v_version == 'large':
            self.wav_2_vec = torchaudio.pipelines.WAV2VEC2_LARGE.get_model()
            input_dim = 1024

        for param in self.wav_2_vec.parameters():
            param.requires_grad = True
    
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, out_features)
        self.use_sigmoid = use_sigmoid
    def forward(self, x):
        #x = x.squeeze(-2)
        x, _ = self.wav_2_vec(x)
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        x = self.layer4(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x

def predict_count(model, features, threshold=0.5, method='schmitt', smooth=True, sliding_window=False, window_size=16000*10):
    if features.dim() == 1:
        features = features.unsqueeze(0)

    features = features.to(device)
    model.eval()

    L = features.shape[-1]
    total_output_len = int((L / 160000) * 499)

    with torch.inference_mode():
        if L <= window_size or not sliding_window:
            heatmap = model(features).squeeze() 
        else:
            #SLIDING WINDOW METHOD!!!
            stride = window_size // 2
            
            full_heatmap = torch.zeros(total_output_len, device=device)
            overlap_count = torch.zeros(total_output_len, device=device)
            
            starts = list(range(0, L - window_size + 1, stride))
            
            # deal with strange lengths
            if len(starts) == 0 or starts[-1] + window_size < L:
                starts.append(L - window_size)
                
            for start in starts:
                end = start + window_size
                chunk = features[..., start:end]
                
                chunk_heatmap = model(chunk).squeeze()

                out_start = int(start * total_output_len / L)
                out_end = out_start + 499
                
                if out_end > total_output_len:
                    out_end = total_output_len
                    chunk_heatmap = chunk_heatmap[:(out_end - out_start)]

                full_heatmap[out_start:out_end] += chunk_heatmap
                overlap_count[out_start:out_end] += 1.0
                
            heatmap = full_heatmap / overlap_count
    
    if smooth:
        sigma = 2.0
        kernel_size = int(sigma * 4 + 1)
        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        kernel = torch.exp(-x.pow(2) / (2 * sigma**2))
        kernel = (kernel / kernel.sum()).view(1, 1, -1).to(heatmap.device)
        heatmap = F.conv1d(heatmap.view(1, 1, -1), kernel, padding=kernel_size//2).squeeze()

    # alternative methods to predict count from heatmap
    # schmitt trigger is preferred
    if method == 'transitions':
        binary_map = (heatmap > threshold).int()
        transitions = (binary_map[1:] > binary_map[:-1]).sum().item()
        if binary_map[0] == 1:
            transitions += 1
        prediction = transitions
    elif method == 'scipy_peaks':
        heatmap_np = heatmap.cpu().numpy()
        peaks, _ = find_peaks(heatmap_np, height=threshold)
        prediction = len(peaks)
    elif method == "schmitt":
        count = 0
        high_thresh = 0.6
        low_thresh = 0.4
        in_event = False
        
        for value in heatmap:
            if not in_event and value > high_thresh:
                # start of a peak
                in_event = True
                count += 1
            elif in_event and value < low_thresh:
                # end of a peak
                in_event = False
        prediction = count
    else:
        raise ValueError(f"Unknown method: {method}")

    return torch.tensor([[float(prediction)]], device=device), heatmap.cpu().numpy()

class AudioWithCSV(torch.utils.data.Dataset):
    def __init__(self, input_dir, pitch_shift=False,normalise=True, csv_path=None, use_w2v=False, cut_start=True, volume_shift=False):
        self.input_dir = input_dir
        self.pitch_shift = pitch_shift
        self.normalise = normalise
        self.use_w2v = use_w2v
        self.csv_data = csv.DictReader(open(csv_path))
        self.csv_dict = {row['location']: int(row['repetitions']) for row in self.csv_data}
        #if rep not between 0 and 8, remove from csv_dict
        self.csv_dict = {k: v for k, v in self.csv_dict.items() if 0 <= v <= 8}

        self.files = [f for f in os.listdir(input_dir) if f in self.csv_dict]
        self.file_number = len(self.files)

        #get start times if cut_start is true
        self.volume_shift = volume_shift
        self.cut_start = cut_start
        if cut_start:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                self.start_dict = {row['location']: float(row['start_time']) for row in reader}
    
    def __len__(self):
        return self.file_number
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        if self.use_w2v:
            features, audio_length = wav_2_vec(os.path.join(self.input_dir, f'{file_name}'), pitch_shift=self.pitch_shift, normalise=self.normalise, volume_shift=self.volume_shift)

            features = features.squeeze(0)

            if self.cut_start:
                start_time = self.start_dict[file_name]
                #convert start time to feature index
                time_dim_size = features.shape[0]
                start_idx = int(start_time * time_dim_size / audio_length)
                features = features[start_idx:, :]
        else:
            features, sr = torchaudio.load(os.path.join(self.input_dir, f'{file_name}'))
            
            if self.volume_shift:
                features /= features.abs().max() + 1e-7

            if features.shape[0] > 1:
                features = features.mean(dim=0, keepdim=True)

            if sr != 16000:
                features = torchaudio.functional.resample(features, sr, 16000)

            if self.pitch_shift:
                features = torchaudio.functional.pitch_shift(features, 16000, n_steps=36)
            
            if self.normalise:
                features = (features - features.mean()) / (features.std() + 1e-7)
            
            features = features.squeeze(0) 

            audio_length = features.shape[0] / 16000
            
            if self.cut_start:
                start_time = self.start_dict[file_name]
                features = features[int(start_time * 16000):]

        label = self.csv_dict[file_name]
        return features, label, torch.tensor([0])

def get_heatmap_quality(heatmap):
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)
    
    heatmap = heatmap.to(device)
    events = []
    
    high_thresh = 0.6
    low_thresh = 0.4
    in_event = False
    start_idx = 0
    
    # recalculate schmitt trigger events
    for i, value in enumerate(heatmap):
        if not in_event and value > high_thresh:
            in_event = True
            start_idx = i
        elif in_event and value < low_thresh:
            in_event = False
            events.append((start_idx, i))

    if len(events) == 0:
        error = torch.mean(heatmap**2).item() 
        return error, np.zeros_like(heatmap.cpu().numpy()), []
    
    x = torch.arange(len(heatmap)).to(device)
    ideal_heatmap = torch.zeros_like(heatmap)
    
    for start, end in events:
        center = (start + end) / 2
        duration = end - start
        
        sigma = duration / 4.0 
        
        ideal_heatmap += torch.exp(-((x - center)**2) / (2 * sigma**2 + 1e-6))
    
    ideal_heatmap = torch.clamp(ideal_heatmap, 0, 1)
    error = F.mse_loss(heatmap, ideal_heatmap).item()
    
    # Extract peak centers for the return value
    peaks_indices = [(s + e) // 2 for s, e in events]
    
    return error, ideal_heatmap.cpu().numpy(), peaks_indices

def test_model(model_name, test_set_name, txt_path, normalise=True, volume_shift=False, measure_mse=False):
    print(f"Testing model: {model_name} on test set: {test_set_name}")

    use_sigmoid = 'NS' not in model_name
    if 'LARGE' in model_name:
        w2v_version = 'large'
    else:
        w2v_version = 'base'

    if "-F" in model_name:
        model = W2V_mlp(out_features=1, use_sigmoid=use_sigmoid, w2v_version=w2v_version)
    else:
        model = transformer_blocks(in_features=768, out_features=1)
    model.load_state_dict(torch.load(f"/users/hani/AudioCounting/models/{model_name}/model_{model_name}.pth", map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()

    #check if csv dataset
    csv_test_sets = {"/scratch/local/hdd/hani/bbc_clocks/audio/": "/users/hani/AudioCounting/preprocessing/bbc_clocks/bbc_clocks.csv"
                     , "/scratch/local/hdd/hani/heartbeats/wav/": "/users/hani/AudioCounting/preprocessing/heartbeats/heartbeats_sorted.csv"
                     , "/scratch/local/hdd/hani/dolphins/test/": "/users/hani/AudioCounting/preprocessing/dolphins/dolphins.csv"
                     , "/scratch/local/hdd/hani/dolphins/test_padded/": "/users/hani/AudioCounting/preprocessing/dolphins/dolphins.csv"}

    csv_path = csv_test_sets.get(test_set_name, None)

    if csv_path == "/users/hani/AudioCounting/preprocessing/bbc_clocks/bbc_clocks.csv":
        cut_start = True
    else:
        cut_start = False

    if "dolphin" in test_set_name:
        volume_shift = True
    else:
        volume_shift = False

    if "-T" in model_name:
        use_w2v = True
    else:
        use_w2v = False

    #get correct dataloader
    if csv_path is None:
        #json dataset
        test_loader = torch.utils.data.DataLoader(
            AudioWithJSON(test_set_name, use_w2v=use_w2v, normalise=normalise),
            batch_size=1, shuffle=False)

    else:
        test_loader = torch.utils.data.DataLoader(
            AudioWithCSV(test_set_name, use_w2v=use_w2v, csv_path=csv_path, normalise=normalise, cut_start=cut_start, volume_shift=volume_shift),
            batch_size=1, shuffle=False)



    # active_methods = [
    #     ('transitions', False), ('transitions', True),
    #     ('scipy_peaks', False), ('scipy_peaks', True),
    #     ('schmitt', False),     ('schmitt', True)
    # ]

    active_methods = [('schmitt', False), ('schmitt', True)] #blur and no blur WavCounter variants

    # active_methods = [('schmitt', True)]

    stats = {f"{m}_{'S' if s else 'R'}": {"mae": 0.0, "corr": 0, "obo": 0} for m, s in active_methods}

    if measure_mse:
        dataset_short_name = test_set_name.strip('/').split('/')[-3]
        csv_log_path = f"/users/hani/AudioCounting/results/MSE_{model_name}_{dataset_short_name}.csv"
        
        csv_file = open(csv_log_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["file", "mse", "ground_truth", "prediction", "is_correct"])

    pbar = tqdm(test_loader, desc=f"Testing {model_name} on {test_set_name}", unit="sample")
    for i, (data, target, _) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        target_val = int(target.item())

        fname = test_loader.dataset.files[i] if hasattr(test_loader.dataset, 'files') else test_loader.dataset.file_names[i]

        for method, smooth in active_methods:
            key = f"{method}_{'S' if smooth else 'R'}"
            
            
            with torch.inference_mode():
                if measure_mse:
                    # get heatmap out for mse
                    output, heatmap = predict_count(model, data, threshold=0.5, method=method, smooth=smooth)
                    pred_val = int(output.item())

                    mse_val, _, _ = get_heatmap_quality(heatmap)
                    is_correct = 1 if pred_val == target_val else 0
                    csv_writer.writerow([fname, f"{mse_val:.4f}", target_val, pred_val, is_correct])
                    if i%50 == 0:
                        csv_file.flush()
                else:
                    output, _ = predict_count(model, data, threshold=0.5, method=method, smooth=smooth)
            
            output = output.to(target.device)
            pred = output.round().long()
            
            stats[key]["corr"] += pred.eq(target.view_as(pred)).sum().item()
            stats[key]["obo"] += (torch.abs(pred - target.view_as(pred)) <= 1).sum().item()
            
            denom = target.view_as(pred).float() + 0.1
            stats[key]["mae"] += (torch.abs(pred - target.view_as(pred)).float() / denom).sum().item()

    if measure_mse:
        csv_file.close()


    n = len(test_loader.dataset)
    print(f"\n{'Method':<20} | {'MAE':<8} | {'Acc %':<8} | {'OBO %':<8}")
    print("-" * 50)

    for key, val in stats.items():
        mae = val["mae"] / n
        obo = 100. * val["obo"] / n
        acc = 100. * val["corr"] / n
        print(f"{key:<20} | {mae:<8.4f} | {acc:<8.2f} | {obo:<8.2f}")

    #save results in txt file
    with open(txt_path, "a") as f:
        f.write(f"\nModel: {model_name} | Dataset: {test_set_name}\n")
        f.write(f"{'Method':<20} | {'MAE':<8} | {'Acc %':<8} | {'OBO %':<8}\n")
        f.write("-" * 50 + "\n")
        for key, val in stats.items():
            mae = val["mae"] / n
            obo = 100. * val["obo"] / n
            acc = 100. * val["corr"] / n
            f.write(f"{key:<20} | {mae:<8.4f} | {acc:<8.2f} | {obo:<8.2f}\n")
        f.write("-" * 50 + "\n")

def test_models_on_all_sets(models, except_RSN=False):
    test_sets = [
        '/scratch/local/hdd/hani/bbc_clocks/audio/',
        '/scratch/local/hdd/hani/heartbeats/wav/',
        '/scratch/local/hdd/hani/dolphins/test_padded/',
        '/scratch/local/ssd/hani/RS/wav/test/',
        '/scratch/local/ssd/hani/RSN/wav/test/',
        '/scratch/local/ssd/hani/RVN/wav/test/',
    ]

    if except_RSN:
        test_sets = [s for s in test_sets if "RSN" not in s]
    
    for model_name in models:

        if not os.path.exists(f"/users/hani/AudioCounting/models/{model_name}/model_{model_name}.pth"):
            print(f"Model file not found for {model_name}")
            continue

        normalise = "LARGE" in model_name

        if normalise:
            txt_path = f"/users/hani/AudioCounting/results_new/{model_name}_n_results.txt"
        else:
            txt_path = f"/users/hani/AudioCounting/results_new/{model_name}_results.txt"

        # if os.path.exists(txt_path):
        #     os.remove(txt_path)

        for test_set in test_sets:
            test_model(model_name, test_set, txt_path, normalise=normalise)

models = ["LARGE-RS-W2V-F", "LARGE-RSN-W2V-F", "LARGE-RVN-W2V-F"] #True WavCounter

test_models_on_all_sets(models)