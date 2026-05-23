# Imports
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import datasets
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import json
import csv
from scipy.signal import find_peaks
import math
import os
import preprocessing.data_converter


# setup
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
wav_2_vec_model = torchaudio.pipelines.WAV2VEC2_BASE.get_model().to(device)
print(f'Using device: {device}')

transform = T.Resize((504, 504), interpolation=T.InterpolationMode.BICUBIC)


finetune = True #set to true to test finetuned w2v model, false to test transformer variant trained on w2v features

if not finetune:

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

#dataset classes

class AudioWithJSON(torch.utils.data.Dataset):
    def __init__(self, input_dir, use_w2v=False, normalise=False):
        self.input_dir = input_dir
        self.file_number = len(os.listdir(input_dir)) - 1
        self.use_w2v = use_w2v
        self.normalise = normalise

        with open(os.path.join(input_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
    
    def __len__(self):
        return self.file_number
    
    def __getitem__(self, idx):
        idx += 1
        file_name = f'{idx:06d}'
        if self.use_w2v:
            features, audio_length = wav_2_vec(os.path.join(self.input_dir, f'{file_name}.wav'), normalise=self.normalise)
            feat_length = features.shape[1]
        else:
            #use wav directly
            features, sr = torchaudio.load(os.path.join(self.input_dir, f'{file_name}.wav'))
            if sr != 16000:
                features = torchaudio.functional.resample(features, sr, 16000)
            if self.normalise:
                features = (features - features.mean()) / (features.std() + 1e-7)
            audio_length = features.shape[1] / 16000
            feat_length = 499

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

def predict_count(model, features, threshold=0.5, method='transitions', smooth=False, sliding_window=True, window_size=16000*10):
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
    def __init__(self, input_dir, pitch_shift=False,normalise=True, csv_path=None, use_w2v=False, cut_start=False, volume_shift=False):
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


class TSSMfromWav(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_path, transform=transform, cut_start=False, volume_shift=False):
        self.data_dir = root_dir
        self.transform = transform
        self.cut_start = cut_start
        self.volume_shift = volume_shift

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            self.csv_dict = {row["location"]: row for row in reader 
                             if 0 <= int(row["repetitions"]) <= 8}

        self.files = sorted([
            f for f in os.listdir(self.data_dir)
            if f in self.csv_dict and f.endswith(".wav") and os.path.getsize(os.path.join(self.data_dir, f)) > 0
        ])

        self.converter = preprocessing.data_converter.DataConverter()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        row = self.csv_dict[file_name]
        label = int(row['repetitions'])
        if self.cut_start:
            start_time = float(row['start_time'])

        audio, sr = torchaudio.load(os.path.join(self.data_dir, file_name))
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
            sr = 16000
        
        audio = torch.mean(audio, dim=0) if audio.shape[0] > 1 else audio.squeeze(0)

        if self.volume_shift:
            audio /= audio.abs().max() + 1e-7
        
        if self.cut_start:
            audio = audio[int(start_time * sr):]
        
        audio = audio.to(device)

        with torch.no_grad():
            features, _ = wav_2_vec_model(audio.unsqueeze(0))
            x = features[0]
            x = F.normalize(x, p=2, dim=-1)
            tssm = (x @ x.T).cpu().numpy()

        tssm = self.converter.apply_histogram_equalisation(tssm)
        
        tssm_tensor = torch.tensor(tssm, dtype=torch.float32).unsqueeze(0) 

        if self.transform:
            tssm = self.transform(tssm_tensor) 
            
        tssm = tssm.repeat(3, 1, 1)
        
        return tssm, label

class TSSMTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, use_csv=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_csv = use_csv
        wav_dir = self.root_dir.replace("tssm", "wav")
        with open(os.path.join(wav_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        self.fnames = sorted(list(self.metadata.keys()))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        arr = np.load(os.path.join(self.root_dir, fname + ".npy"))
        if arr.ndim == 2: arr = np.expand_dims(arr, axis=0)
        if arr.shape[0] == 1: arr = np.repeat(arr, 3, axis=0)
        arr = arr.astype(np.float32) / (arr.max() + 1e-8)
        
        tensor = torch.tensor(arr, dtype=torch.float32)
        if self.transform: 
            tensor = self.transform(tensor)
        
        label = self.metadata[fname]['num_repetitions']
        return tensor, label


#model definitions

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
        x, _ = self.wav_2_vec(x)
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        x = self.layer4(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x


class DinoFullWithClassifier(nn.Module):
    def __init__(self, dino_model, num_classes):
        super().__init__()
        self.dino = dino_model

        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Freeze all blocks except the first 4
        for i, blk in enumerate(self.dino.blocks):
            if i >= 4:
                for param in blk.parameters():
                    param.requires_grad = False

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.dino.patch_embed(x)

        cls = self.dino.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        pos = self.interpolate_pos_encoding(x, H, W)
        x = x + pos

        for i, blk in enumerate(self.dino.blocks):
            x = blk(x)

        x = self.dino.norm(x)
        x = x[:, 0]  #CLS

        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)

    #From https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
    #allows dinov1 to handle different input sizes by interpolating pos embeddings
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.dino.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.dino.pos_embed
        class_pos_embed = self.dino.pos_embed[:, 0]
        patch_pos_embed = self.dino.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.dino.patch_embed.patch_size
        h0 = h // self.dino.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

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
        return 1.0, heatmap.cpu().numpy(), []
    
    x = torch.arange(len(heatmap)).to(device)
    ideal_heatmap = torch.zeros_like(heatmap)
    
    for start, end in events:
        #place scaled gaussian at center of each event
        center = (start + end) / 2
        duration = end - start
        
        sigma = duration / 4.0 
        
        ideal_heatmap += torch.exp(-((x - center)**2) / (2 * sigma**2 + 1e-6))
    
    #find mse between "ideal" and actual heatmap
    ideal_heatmap = torch.clamp(ideal_heatmap, 0, 1)
    error = F.mse_loss(heatmap, ideal_heatmap).item()
    
    peaks_indices = [(s + e) // 2 for s, e in events]
    
    return error, ideal_heatmap.cpu().numpy(), peaks_indices

#threshold of 0.0640 separates heatmap confidences of correct vs incorrect predictions
def test_combined(test_set_path, wav_model_name, tssm_model_name, csv_path, threshold=0.0640, txt_path = None):

    print(f"Testing with WAV model: {wav_model_name} and TSSM model: {tssm_model_name}")
    output_dir = "/users/hani/AudioCounting/models/"

    #WAVCOUNTER MODEL

    if "LARGE" in wav_model_name:
        w2v_version = 'large'
    else:
        w2v_version = 'base'

    if "NS" in wav_model_name:
        use_sigmoid = False
    else:
        use_sigmoid = True

    wav_model = W2V_mlp(out_features=1, use_sigmoid=use_sigmoid, w2v_version=w2v_version)
    wav_model.load_state_dict(torch.load(os.path.join(output_dir, wav_model_name, f"model_{wav_model_name}.pth"), map_location=device))
    wav_model = wav_model.to(device)
    wav_model.eval()

    #TSSMCOUNTER MODEL

    vit8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    tssm_model = DinoFullWithClassifier(vit8, num_classes=9).to(device)
    load_path = f"/users/hani/AudioCounting/models/{tssm_model_name}/model_{tssm_model_name}.pth"
    tssm_model.load_state_dict(torch.load(load_path, map_location=device))
    tssm_model.eval()


    use_w2v = "-T" in wav_model_name 
    cut_start = "clocks" in test_set_path
    normalise = "LARGE" in wav_model_name
    volume_shift = "dolphins" in test_set_path

    if csv_path is None:
        wav_dataset = AudioWithJSON(test_set_path, use_w2v=use_w2v, normalise=normalise)
        tssm_loader = torch.utils.data.DataLoader(
            TSSMTestDataset(test_set_path.replace("wav", "tssm"), transform=transform),
            batch_size=1, shuffle=False
        )
    else:
        wav_dataset = AudioWithCSV(test_set_path, csv_path=csv_path, use_w2v=use_w2v, cut_start=cut_start, normalise=normalise, volume_shift=volume_shift)
        wav_dataset.files = sorted(wav_dataset.files)
        tssm_loader = torch.utils.data.DataLoader(
            TSSMfromWav(test_set_path, csv_path=csv_path, transform=transform, cut_start=cut_start, volume_shift=volume_shift),
            batch_size=1, shuffle=False
        )
    
    wav_loader = torch.utils.data.DataLoader(wav_dataset, batch_size=1, shuffle=False)

    stats = {"mae": 0.0, "corr": 0, "obo": 0}
    n = len(wav_loader.dataset)

    number_of_wavcounter_trusts = 0
    number_of_tssmcounter_trusts = 0

    pbar = tqdm(zip(wav_loader, tssm_loader), total=n)

    for i, ((wav_data, target, _), (tssm_data, _)) in enumerate(pbar):
        wav_data = wav_data.to(device)
        tssm_data = tssm_data.to(device)
        target = target.to(device)
        
        with torch.inference_mode():
            wav_pred, heatmap = predict_count(wav_model, wav_data, method='schmitt', smooth=True, sliding_window=False)

            quality_score, ideal_heatmap, peaks = get_heatmap_quality(torch.tensor(heatmap))

            if quality_score < threshold:
                final_count = wav_pred.item()
                number_of_wavcounter_trusts += 1
            else:
                tssm_pred = tssm_model(tssm_data).max(1)[1].float()
                final_count = tssm_pred.item()
                number_of_tssmcounter_trusts += 1

            wav_prop = (number_of_wavcounter_trusts / (i + 1)) * 100
            pbar.set_postfix({"WavTrust": f"{wav_prop:.2f}%"})

            pred = torch.tensor([[final_count]], dtype=torch.long, device=device)

        stats["corr"] += pred.eq(target.view_as(pred)).sum().item()
        stats["obo"] += (torch.abs(pred - target.view_as(pred)) <= 1).sum().item()
        
        denom = target.view_as(pred).float() + 0.1
        stats["mae"] += (torch.abs(pred - target.view_as(pred)).float() / denom).sum().item()

    mae = stats["mae"] / n
    obo = 100. * stats["obo"] / n
    acc = 100. * stats["corr"] / n
    
    print(f"\n MAE: {mae:.4f}| Acc: {acc:.2f}% | OBO: {obo:.2f}%")
    print(f"Number of WavCounter Trusts: {number_of_wavcounter_trusts}")
    print(f"Number of TSSMCounter Trusts: {number_of_tssmcounter_trusts}")

    if txt_path is not None:
        txt_path = os.path.join("/users/hani/AudioCounting/results/", txt_path)
        with open(txt_path, "a") as f:
            f.write(f"\nResults for WAV model: {wav_model_name} and TSSM model: {tssm_model_name} on {test_set_path}\n")
            f.write(f"MAE: {mae:.4f} | Acc: {acc:.2f}% | OBO: {obo:.2f}%\n")
            f.write(f"Number of WavCounter Trusts: {number_of_wavcounter_trusts}\n")
            f.write(f"Number of TSSMCounter Trusts: {number_of_tssmcounter_trusts}\n")

test_sets = [
                '/scratch/local/ssd/hani/RS/wav/test/',
                '/scratch/local/ssd/hani/RSN/wav/test/',
                '/scratch/local/ssd/hani/RVN/wav/test/',
                '/scratch/local/hdd/hani/dolphins/test_padded/',
                '/scratch/local/hdd/hani/heartbeats/wav/',
                '/scratch/local/hdd/hani/bbc_clocks/audio/'
            ]

csv_paths = [
                None,
                None,
                None,
                "/users/hani/AudioCounting/preprocessing/dolphins/dolphins.csv",
                "/users/hani/AudioCounting/preprocessing/heartbeats/heartbeats_sorted.csv",
                "/users/hani/AudioCounting/preprocessing/bbc_clocks/bbc_clocks.csv"
            ]

model_combinations = [
                        # ("LARGE-RS-W2V-F", "RS-C"),
                        # ("LARGE-RSN-W2V-F", "RSN-C"),
                        ("LARGE-RVN-W2V-F", "RSN-C")
                        ]

for (wav_model_name, tssm_model_name) in model_combinations:
    log_file = f"/users/hani/AudioCounting/results_new/DUAL_{wav_model_name}_{tssm_model_name}.txt"
    for test_set_path, csv_path in zip(test_sets, csv_paths):
        with open(log_file, "a") as f:
             f.write(f"Testing on {test_set_path} with CSV: {csv_path}\n")
             f.write(f"WAV Model: {wav_model_name}, TSSM Model: {tssm_model_name}\n")

        #don't let one failed test stop the others from running
        try:
            test_combined(test_set_path, wav_model_name, tssm_model_name, csv_path, txt_path=log_file)
        except Exception as e:
            print(f"Error testing {wav_model_name} + {tssm_model_name} on {test_set_path}: {e}")
