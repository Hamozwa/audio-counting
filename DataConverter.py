import torchaudio
import torch
import random
from scipy import signal
import numpy as np
import cv2
import matplotlib.pyplot as plt

output_time = 16
max_repetitions = 8 #lowered if clips cannot fit into this time

def create_augmented_wav(file, output_time, max_repetitions):
    #ignore non-wav files
    if not file.endswith(".wav"):
        return

    y, sample_rate = torchaudio.load(file, normalize=True)
    file_samples = y.shape[1]
    target_samples = output_time * sample_rate

    if file_samples > target_samples:
        print(file + " is too long")
        return

    #Decide rep number and resulting zeroes length to total desired output time
    max_repetitions = min(round(target_samples//file_samples), max_repetitions)
    num_repetitions = random.randint(0,max_repetitions)
    num_repetitions = max_repetitions
    print(num_repetitions, "repetitions")
    zeroes_samples = target_samples - num_repetitions*file_samples

    #Randomly split zeroes lengths around repetitions
    cuts = sorted([random.randint(0, zeroes_samples) for _ in range(num_repetitions)])
    points = [0] + cuts + [zeroes_samples]
    zeroes_lengths = []
    for j in range(num_repetitions):
        zeroes_lengths.append(points[j+1]-points[j])

    output = torch.zeros(y.shape[0], target_samples)
    ptr = 0
    for i in range(num_repetitions):
        ptr += zeroes_lengths[i]
        output[:, ptr:ptr+file_samples] = y
        ptr += file_samples

    return output, sample_rate

def create_spectrogram_npy(y, sr, out_npy_path, nperseg=512, noverlap=353) -> None:
    y = y + torch.empty_like(y).uniform_(-0.01,0.01)

    waveform = y.mean(dim=0).cpu().numpy()
    _, _, spectrogram = signal.spectrogram(waveform, sr, nperseg=nperseg, noverlap=noverlap)
    spectrogram = np.log(spectrogram + 1e-7)

    #normalise
    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    spectrogram = np.divide(spectrogram - mean, std + 1e-9)

    # noise = np.random.uniform(-0.05,0.05, spectrogram.shape)
    # spectrogram = spectrogram + noise

    spectrogram = cv2.resize(spectrogram, (224, 224))

    # save spectrogram as float32
    np.save(out_npy_path, spectrogram.astype(np.float32))

def create_mel_spectrogram_npy(y, sr, out_npy_path, n_mels=224, n_fft=512, hop_length=160, win_length=400) -> None:
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels
    )
    mel_spectrogram = mel_spectrogram_transform(y)
    mel_spectrogram = torch.log(mel_spectrogram + 1e-7)

    # normalise
    mean = mel_spectrogram.mean()
    std = mel_spectrogram.std()
    mel_spectrogram = (mel_spectrogram - mean) / (std + 1e-9)

    mel_spectrogram = mel_spectrogram.squeeze().cpu().numpy()
    mel_spectrogram = cv2.resize(mel_spectrogram, (224, 224))

    # save spectrogram as float32
    np.save(out_npy_path, mel_spectrogram.astype(np.float32))

def visualize_npy(npy_path, out_png, cmap="magma"):
    a = np.load(npy_path)

    # account for channels
    if a.ndim == 3:
        a = a.mean(axis=0)
    
    print("min,max,shape:", float(a.min()), float(a.max()), a.shape)

    plt.figure(figsize=(224/150, 224/150))  # width, height in inches = pixels / DPI
    plt.imshow(a, origin='lower', aspect='equal', cmap=cmap)
    plt.axis('off')
    plt.savefig(out_png, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()


# y, sr = create_augmented_wav("drop.wav", output_time, max_repetitions)
# print(y.shape, sr, y.abs().max())
# create_spectrogram(y, sr)

y, sr = create_augmented_wav("drop.wav", output_time, max_repetitions)
create_spectrogram_npy(y, sr, "drop_spec.npy")
visualize_npy("drop_spec.npy", "drop_spec_viz.png")