import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import json
import time
from tqdm import tqdm

import os

np.random.seed(0)
torch.manual_seed(0)

input_dir = '/scratch/local/ssd/hani/RS/wav/'
model_name = 'LARGE-RS-W2V-F'
output_dir = '/users/hani/AudioCounting/models'
LOSS = 'mse' #mse or bce

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if "-F-" not in model_name:
    #Transformer variant
    wav_2_vec_bundle = torchaudio.pipelines.WAV2VEC2_BASE
    wav_2_vec_model = wav_2_vec_bundle.get_model()
    wav_2_vec_model = wav_2_vec_model.to(device)

    def standardize_audio(waveform):
        mu = waveform.mean()
        sigma = waveform.std()
        return (waveform - mu) / (sigma + 1e-7)

    def wav_2_vec(file, normalise=False):
        waveform, sr = torchaudio.load(file)
        #make mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        length = waveform.shape[1]/sr
        waveform = waveform.to(device)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if normalise:
            waveform = standardize_audio(waveform)
        with torch.no_grad():
            features, _ = wav_2_vec_model(waveform)
        return features, length

if "LARGE" in model_name:
    print("Using wav2vec LARGE model => using normalisation")
    normalise = True
else:
    normalise = False

if not os.path.exists(os.path.join(output_dir, model_name)):
    print("Output dir does not exist! Creating it.")
    os.makedirs(os.path.join(output_dir, model_name))
    if not os.path.exists(os.path.join(output_dir, model_name)):
        raise Exception(f"Failed to create output directory {os.path.join(output_dir, model_name)}")
    else:
        print(f"Successfully created output directory {os.path.join(output_dir, model_name)}")
else:
    print("Output dir exists :)")

class AudioWithJSON(torch.utils.data.Dataset):
    def __init__(self, input_dir, normalise=True ,use_w2v=True):
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
            
        return features[0], label, target
    

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

model = W2V_mlp(out_features=1, use_sigmoid=False, w2v_version='large')
model = model.to(device)

def train(epoch):
    sum_loss = 0.0
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True, unit="batch")
    script_start = time.time()

    for batch_idx, (data, label, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        if batch_idx % accum_steps == 0:
            optimizer.zero_grad()
        
        output = model(data).squeeze(-1)

        if LOSS == 'bce':
            loss = criterion(output, target) / accum_steps
        elif LOSS == 'mse':
            weight = (target > 0.1).float() * 10.0 + 1.0
            loss = (weight * (output - target)**2).sum() / accum_steps

        sum_loss += loss.item() * accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()

        total_elapsed = (time.time() - script_start) / 60  # in minutes
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accum_steps:.4f}',
            'total_min': f'{total_elapsed:.1f}'
        })
    return sum_loss / len(train_loader.dataset)

def val():
    model.eval()
    val_loss = 0
    correct_counts = 0
    obo_counts = 0
    total_samples = len(val_loader.dataset)

    for data, label, target in val_loader:
        data, target = data.to(device), target.to(device)
        
        with torch.inference_mode():
            output = model(data).squeeze(-1)
            # if LOSS == 'bce':
            #     output = torch.sigmoid(output)
        
        if LOSS == "mse":
            val_loss += F.mse_loss(output, target, reduction='sum').item()
        elif LOSS == "bce":
            val_loss += F.binary_cross_entropy(output, target, reduction='sum').item()

        binary = (output > 0.5).int()
        pred_count = (binary[:, 1:] > binary[:, :-1]).sum(dim=1)
        pred_count += binary[:, 0] 

        label = label.to(device)
        correct_counts += (pred_count == label).sum().item()
        
        obo_counts += (torch.abs(pred_count - label) <= 1).sum().item()

    avg_loss = val_loss / total_samples
    acc = 100. * correct_counts / total_samples
    obo_acc = 100. * obo_counts / total_samples

    print(f'\nVal Set: Avg Heatmap Loss: {avg_loss:.4f}')
    print(f'Count Accuracy: {correct_counts}/{total_samples} ({acc:.2f}%)')
    print(f'OBO Accuracy: {obo_acc:.2f}%\n')
    
    return avg_loss, acc, obo_acc

batch_size = 8
learning_rate = 0.0001
epochs = 30
accum_steps = 8 #effective batch size = batch_size * accum_steps

# # one dataset
train_loader = torch.utils.data.DataLoader(
    AudioWithJSON(input_dir + 'train/', use_w2v=False, normalise=normalise),
    batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    AudioWithJSON(input_dir + 'val/', use_w2v=False, normalise=normalise),
    batch_size=1, shuffle=False)

# Multiple datasets

# RS_train_data = AudioWithJSON('/scratch/local/ssd/hani/RS/wav/train/', use_w2v=False, normalise=normalise)
# RSN_train_data = AudioWithJSON('/scratch/local/ssd/hani/RSN/wav/train/', use_w2v=False, normalise=normalise)
# RVN_train_data = AudioWithJSON('/scratch/local/ssd/hani/RVN/wav/train/', use_w2v=False, normalise=normalise)

# train_loader = torch.utils.data.DataLoader(
#     torch.utils.data.ConcatDataset([RS_train_data,RSN_train_data, RVN_train_data]),
#     batch_size=batch_size, shuffle=True)

# RS_val_data = AudioWithJSON('/scratch/local/ssd/hani/RS/wav/val/', use_w2v=False, normalise=normalise)
# RSN_val_data = AudioWithJSON('/scratch/local/ssd/hani/RSN/wav/val/', use_w2v=False, normalise=normalise)
# RVN_val_data = AudioWithJSON('/scratch/local/ssd/hani/RVN/wav/val/', use_w2v=False, normalise=normalise)

# val_loader = torch.utils.data.DataLoader(
#     torch.utils.data.ConcatDataset([RS_val_data, RSN_val_data, RVN_val_data]),
#     batch_size=1, shuffle=False)


if "LARGE" in model_name:
    w2v_params = list(model.wav_2_vec.parameters())
    mlp_params = [p for n, p in model.named_parameters() if n.startswith('layer')]

    optimizer = torch.optim.Adam([
        {'params': model.wav_2_vec.feature_extractor.parameters(), 'lr': 1e-6}, 
        {'params': model.wav_2_vec.encoder.transformer.layers[:12].parameters(), 'lr': 1e-6}, 
        {'params': model.wav_2_vec.encoder.transformer.layers[12:].parameters(), 'lr': 1e-5}, 
        {'params': mlp_params, 'lr': 1e-4} 
    ])
else:
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

if LOSS == 'bce':
    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    criterion = nn.BCELoss()

print(f"Training model: {model_name}")

start_training_time = time.time()

best_obo_acc = -1.0

epoch_list = []
train_losses = []
val_losses = []
val_accs = []
val_obo_accs = []
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()

    epoch_list.append(epoch)
    train_loss = train(epoch)
    train_losses.append(train_loss)
    val_loss, val_acc, val_obo_acc = val()

    if val_obo_acc > best_obo_acc:
        best_obo_acc = val_obo_acc
        torch.save(model.state_dict(), os.path.join(output_dir, model_name, f"model_{model_name}.pth"))
        print(f"Saved new best OBO model: {best_obo_acc:.2f}% (epoch {epoch})")

    epoch_duration = time.time() - epoch_start_time
    total_elapsed = time.time() - start_training_time

    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_obo_accs.append(val_obo_acc)

    print(f"Training model {model_name}, Epoch {epoch}/{epochs}")
    print(f"Epoch {epoch} completed in {epoch_duration:.2f}s")
    print(f"Total time elapsed: {total_elapsed/60:.2f} minutes")
    print(f"Expected remaining time: {(total_elapsed/epoch) * (epochs - epoch)/60:.2f} minutes")

total_time = time.time() - start_training_time
print(f"Training model {model_name} complete, best OBO accuracy: {best_obo_acc:.2f}%")
print(f"Total Training Time: {total_time/60:.2f} minutes")