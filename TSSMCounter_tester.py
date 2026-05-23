import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import math
import csv
import preprocessing.data_converter
import torchaudio
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = T.Resize((504, 504), interpolation=T.InterpolationMode.BICUBIC)

import torch.nn.functional as FT
wav_2_vec_bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav_2_vec_model = wav_2_vec_bundle.get_model().to(device)

# MODEL CLASS --------------------------------------------------------
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
        
        # Freeze all blocks except the first 4 (finetune to extract tssm features)
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
            if i >= 4:
                with torch.no_grad():  # prevents storing activations for frozen blocks
                    x = blk(x)
            else:
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

# Dataset Classes --------------------------------------------------------

class TSSMTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        wav_dir = self.root_dir.replace("tssm", "wav")
        with open(os.path.join(wav_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        self.fnames = sorted(list(self.metadata.keys()))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        arr = np.load(os.path.join(self.root_dir, fname + ".npy"))
        if arr.ndim == 2: 
            arr = np.expand_dims(arr, axis=0)
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        arr = arr.astype(np.float32) / (arr.max() + 1e-8)
        
        tensor = torch.tensor(arr, dtype=torch.float32)
        if self.transform:
            tensor = self.transform(tensor)
        
        label = self.metadata[fname]['num_repetitions']
        return tensor, label

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
            x = FT.normalize(x, p=2, dim=-1)
            tssm = (x @ x.T).cpu().numpy()

        tssm = self.converter.apply_histogram_equalisation(tssm)
        
        tssm_tensor = torch.tensor(tssm, dtype=torch.float32).unsqueeze(0) 

        if self.transform:
            tssm = self.transform(tssm_tensor) 
            
        tssm = tssm.repeat(3, 1, 1)
        
        return tssm, label

# --- Test Function ---
def test_dino_model(model_name, test_set_path):
    print(f"Testing DINO model: {model_name} on {test_set_path}")
    
    #load model
    vit8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    model = DinoFullWithClassifier(vit8, num_classes=9).to(device)
    
    load_path = f"/users/hani/AudioCounting/models/{model_name}/model_{model_name}.pth"
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    #get test loader
    csv_test_sets = {"/scratch/local/hdd/hani/bbc_clocks/audio/": "/users/hani/AudioCounting/preprocessing/bbc_clocks/bbc_clocks.csv"
                     , "/scratch/local/hdd/hani/heartbeats/wav/": "/users/hani/AudioCounting/preprocessing/heartbeats/heartbeats_sorted.csv"
                     , "/scratch/local/hdd/hani/dolphins/test/": "/users/hani/AudioCounting/preprocessing/dolphins/dolphins.csv"
                     , "/scratch/local/hdd/hani/dolphins/test_padded/": "/users/hani/AudioCounting/preprocessing/dolphins/dolphins.csv"}

    csv_path = csv_test_sets.get(test_set_path, None)

    if csv_path:
        volume_shift = "dolphins" in test_set_path
        cut_start = "bbc_clocks" in test_set_path
        test_loader = torch.utils.data.DataLoader(
            TSSMfromWav(test_set_path, csv_path=csv_path, transform=transform, volume_shift=volume_shift, cut_start=cut_start),
            batch_size=1, shuffle=False
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            TSSMTestDataset(test_set_path, transform=transform),
            batch_size=1, shuffle=False
        )


    # test loop
    correct = 0
    obo_correct = 0
    mae_sum = 0.0
    n = len(test_loader.dataset)

    for data, target in tqdm.tqdm(test_loader, desc="Testing"):
        data, target = data.to(device), target.to(device)
        
        with torch.inference_mode():
            output = model(data)
            pred = output.max(1)[1]

        correct += pred.eq(target.view_as(pred)).sum().item()
        obo_correct += (torch.abs(pred - target.view_as(pred)) <= 1).sum().item()
        
        denom = target.float() + 0.1
        mae_sum += (torch.abs(pred - target.view_as(pred)).float() / denom).sum().item()

    acc = 100. * correct / n
    obo = 100. * obo_correct / n
    mae = mae_sum / n

    print(f"Results: MAE: {mae:.4f} | OBO: {obo:.2f}% | Acc: {acc:.2f}%")

    # Save all results to the same results file
    txt_path = "/users/hani/AudioCounting/results_new/TSSMCounter_results.txt"
    with open(txt_path, "a") as f:
        f.write(f"\nModel: {model_name} | Dataset: {test_set_path}\n")
        f.write(f"{'MAE':<8}| {'Acc %':<8} | {'OBO %':<8}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{mae:<8.4f} | {acc:<8.2f} | {obo:<8.2f}\n")
        f.write("-" * 50 + "\n")

def test_model_on_all_sets(model_name):
    test_sets = [
        '/scratch/local/ssd/hani/RS/tssm/test/',
        '/scratch/local/ssd/hani/RSN/tssm/test/',
        '/scratch/local/ssd/hani/RVN/tssm/test/',
        '/scratch/local/hdd/hani/bbc_clocks/audio/',
        '/scratch/local/hdd/hani/heartbeats/wav/',
        '/scratch/local/hdd/hani/dolphins/test_padded/'
    ]

    for test_set in test_sets:
        test_dino_model(model_name, test_set)

models = ['RS-C','RSN-C','RVN-C']

for model in models:
    test_model_on_all_sets(model)