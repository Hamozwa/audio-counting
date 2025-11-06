import torchaudio
import torch
import random
from scipy import signal
import numpy as np
import cv2
import matplotlib.pyplot as plt

output_time = 20
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
    # num_repetitions = max_repetitions
    zeroes_samples = target_samples - num_repetitions*file_samples

    #Randomly split zeroes lengths around repetitions
    cuts = sorted([random.randint(0, zeroes_samples) for _ in range(num_repetitions)])
    points = [0] + cuts + [zeroes_samples]
    zeroes_lengths = []
    for j in range(num_repetitions):
        zeroes_lengths.append(points[j+1]-points[j])

    if num_repetitions > 0:
        # Random start and end padding (~10% mean each)
        start_padding = min(max(0, int(random.gauss(zeroes_samples*0.1, zeroes_samples*0.05))), zeroes_samples*0.5)
        end_padding = min(max(0, int(random.gauss(zeroes_samples*0.1, zeroes_samples*0.05))), zeroes_samples*0.5)

        remaining_zeroes = zeroes_samples - start_padding - end_padding

        zeroes_lengths = [start_padding]
        if num_repetitions == 1:
            end_padding += remaining_zeroes  # all leftover goes after the single repetition
        else:
            # Distribute remaining_zeroes approximately evenly between repetitions
            base_gap = remaining_zeroes // (num_repetitions - 1)
            extra = remaining_zeroes % (num_repetitions - 1)

            for i in range(num_repetitions - 1):
                gap = base_gap
                if i < extra:
                    gap += 1  # distribute leftover
                # small random perturbation around gap
                perturb = int(random.gauss(0, remaining_zeroes*0.01))
                perturb = max(-gap, min(perturb, remaining_zeroes))  # keep valid
                zeroes_lengths.append(gap + perturb)
            
            # Adjust zeroes_lengths to sum exactly remaining_zeroes
            diff = remaining_zeroes - sum(zeroes_lengths)
            end_padding += diff 
            if end_padding < 0:
                zeroes_lengths[0] += end_padding
            if zeroes_lengths[0] < 0:
                zeroes_lengths[-1] += zeroes_lengths[0]
                zeroes_lengths[0] = 0
            #else:
            #    god help you
                
    else:
        start_padding = 0
        end_padding = zeroes_samples
        zeroes_lengths = []

    output = torch.zeros(y.shape[0], target_samples)
    ptr = 0
    for i in range(num_repetitions):
        ptr += zeroes_lengths[i]
        output[:, ptr:ptr+file_samples] = y
        ptr += file_samples

    return output, sample_rate, num_repetitions

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

def batch_create_spectrogram_samples(input_folder, output_folder, output_time, max_repetitions):
    import os
    for file in os.listdir(input_folder):
        if not file.endswith(".wav"):
            continue
        filepath = os.path.join(input_folder, file)
        y, sr, num_repetitions = create_augmented_wav(filepath, output_time, max_repetitions)
        base_name = os.path.splitext(file)[0]
        npy_path = os.path.join(output_folder, base_name + "_spec_" + str(num_repetitions) + ".npy")
        create_spectrogram_npy(y, sr, npy_path)

y, sr, num_repetitions = create_augmented_wav("drop.wav", output_time, max_repetitions)
create_spectrogram_npy(y, sr, "drop_spec_" + str(num_repetitions) + ".npy")
visualize_npy("drop_spec_" + str(num_repetitions) + ".npy", "drop_spec_viz.png")