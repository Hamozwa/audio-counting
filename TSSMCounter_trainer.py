# Import modules
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math

import os

np.random.seed(0)
torch.manual_seed(0)

#define directories and model name
input_dir = '/scratch/local/ssd/hani/RVN/tssm'
model_name = 'RVN-C'
output_dir = '/users/hani/AudioCounting/models'

if not os.path.exists(os.path.join(output_dir, model_name)):
    raise Exception(f"Output directory {os.path.join(output_dir, model_name)} does not exist.")

# Load DINOV1 model
vit8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vit8.to(device)
vit8.train()
print(f"Using device: {device}")

transform = T.Resize((504, 504), interpolation=T.InterpolationMode.BICUBIC)

class TSSMNPYwithJSONDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(os.path.join(self.root_dir.replace("tssm","wav"), 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)

        self.fnames = sorted(list(self.metadata.keys()))
        self.file_no = len(self.metadata)

    def __len__(self):
        return self.file_no

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        arr = np.load(os.path.join(self.root_dir, fname + ".npy"))

        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)

        arr = arr.astype(np.float32)
        arr = arr / (arr.max() + 1e-8)

        tensor = torch.tensor(arr, dtype=torch.float32)
        if self.transform:
            tensor = self.transform(tensor)

        item = self.metadata[fname]
        label = item['num_repetitions']
        return tensor, label

train_data = TSSMNPYwithJSONDataset(input_dir + "/train/", transform=transform)
val_data  = TSSMNPYwithJSONDataset(input_dir + "/val/", transform=transform)

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
        
        #Freeze all blocks except the first 4 (finetune to extract tssm features)
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
    
model = DinoFullWithClassifier(vit8, num_classes=9).to(device)
model.to(device)

#Train and val functions
def train(epoch):
    sum_loss = 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if batch_idx % accum_steps == 0:
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum') / accum_steps
        sum_loss += loss.item()
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
        if batch_idx % 1500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return sum_loss / len(train_loader.dataset)

def val():
    model.eval()
    val_loss = 0
    correct = 0
    obo_correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        with torch.inference_mode():
            output = model(data)
        val_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        obo_correct += (np.abs(pred.cpu().numpy().flatten() -
                       target.cpu().numpy().flatten()) <= 1).sum()


    val_loss /= len(val_loader.dataset)
    
    acc = 100. * correct / len(val_loader.dataset)
    obo_acc = 100. * obo_correct / len(val_loader.dataset)

    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), OBO Accuracy: ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), acc, obo_acc))
    return val_loss, acc, obo_acc

batch_size = 4
learning_rate = 0.0001
epochs = 30
accum_steps = 8 #effective batch size = batch_size * accum_steps

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=1, shuffle=False
)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), #filter out frozen params (for efficiency)
    lr=learning_rate
)

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

    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_obo_accs.append(val_obo_acc)

    epoch_duration = time.time() - epoch_start_time
    total_elapsed = time.time() - start_training_time

    print(f"Epoch {epoch} completed in {epoch_duration:.2f}s")
    print(f"Total time elapsed: {total_elapsed/60:.2f} minutes")
    print(f"Expected remaining time: {(total_elapsed/epoch) * (epochs - epoch)/60:.2f} minutes")

total_time = time.time() - start_training_time
print(f"Training complete, best OBO accuracy: {best_obo_acc:.2f}%")
print(f"Total Training Time: {total_time/60:.2f} minutes")