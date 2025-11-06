import os
import random

import torch
import torchaudio
from scipy import signal
import numpy as np
import cv2
import matplotlib.pyplot as plt

class DataConverter:
    def __init__(self, input_time=2, output_time=20, max_repetitions=8):
        self.input_time = input_time # maximum input audio length in seconds
        self.output_time = output_time
        self.max_repetitions = max_repetitions

        # Pre-scan MUSAN noise directories
        folder = "/scratch/local/ssd/hani/musan/noise/"
        search_dirs = [os.path.join(folder, 'sound-bible'), os.path.join(folder, 'free-sound')]
        wav_files = []
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith('.wav'):
                        wav_files.append(os.path.join(root, f))
        self.musan_folder = folder
        self.musan_wav_files = wav_files
        if not self.musan_wav_files:
            print(f"Warning: no MUSAN .wav files found in {search_dirs}")

    def create_augmented_wav(self, file, output_time, max_repetitions):
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

    def add_musan_noise(self, y, sr, snr_db_range=(0,10), debug_wav=False):
        noise_file = random.choice(self.musan_wav_files)
        noise_y, noise_sr = torchaudio.load(noise_file, normalize=True)

        # Resample noise if needed
        if noise_sr != sr:
            resampler = torchaudio.transforms.Resample(noise_sr, sr)
            noise_y = resampler(noise_y)
            noise_sr = sr

        # Ensure noise fits in the length of y. If shorter, place once; if longer,
        # randomly pick a segment of appropriate length so mixing is varied.
        target_len = y.shape[1]
        n_channels_target = y.shape[0]
        noise_len = noise_y.shape[1]

        if noise_len < target_len:
            # Place the noise once at a random location inside the target length
            noise_stream = torch.zeros(n_channels_target, target_len)
            insert_start = random.randint(0, target_len - noise_len)
            # If noise has single channel, duplicate into all target channels
            if noise_y.shape[0] == n_channels_target:
                noise_stream[:, insert_start:insert_start + noise_len] = noise_y
            elif noise_y.shape[0] == 1 and n_channels_target > 1:
                noise_stream[:, insert_start:insert_start + noise_len] = noise_y.repeat(n_channels_target, 1)
            else:
                mono = noise_y.mean(dim=0, keepdim=True)
                noise_stream[:, insert_start:insert_start + noise_len] = mono.repeat(n_channels_target, 1)
            noise_y = noise_stream
        else:
            start = random.randint(0, noise_len - target_len)
            noise_y = noise_y[:, start:start + target_len]

        # Match channels
        if noise_y.shape[0] != n_channels_target:
            if noise_y.shape[0] == 1 and n_channels_target > 1:
                noise_y = noise_y.repeat(n_channels_target, 1)
            else:
                mono = noise_y.mean(dim=0, keepdim=True)
                noise_y = mono.repeat(n_channels_target, 1)

        # Choose random SNR in dB and scale noise accordingly
        snr_db = random.uniform(snr_db_range[0], snr_db_range[1])

        sig_rms = torch.sqrt(torch.mean(y ** 2))
        noise_rms = torch.sqrt(torch.mean(noise_y ** 2))
        if noise_rms == 0 or sig_rms == 0:
            # Nothing to mix or silent target — return original
            return y

        desired_noise_rms = sig_rms / (10 ** (snr_db / 20.0))
        scale = desired_noise_rms / (noise_rms + 1e-9)
        noise_y = noise_y * scale

        mixed = y + noise_y
        # Keep values in valid range
        mixed = torch.clamp(mixed, -1.0, 1.0)

        if debug_wav:
            torchaudio.save("debug_original.wav", y, sr)
            torchaudio.save("debug_noise.wav", noise_y, sr)
            torchaudio.save("debug_mixed.wav", mixed, sr)

        return mixed

    def create_spectrogram_npy(self, y, sr, out_npy_path, nperseg=512, noverlap=353):
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

    def create_mel_spectrogram_npy(self, y, sr, out_npy_path, n_mels=224, n_fft=512, hop_length=160, win_length=400):
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

    def visualize_npy(self, npy_path, out_png, cmap="magma"):
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

    def batch_create_spectrogram_samples(self, input_folder, output_folder):
        # simple progress counter
        all_files = os.listdir(input_folder)
        total_wavs = sum(1 for f in all_files if f.lower().endswith('.wav'))
        counter = 0
        step = max(1, total_wavs // 10) if total_wavs > 0 else 1

        for file in all_files:
            if not file.endswith(".wav"):
                continue
            counter += 1
            if counter % step == 0 or counter == total_wavs:
                print(f"Processed {counter}/{total_wavs} files...")

            filepath = os.path.join(input_folder, file)
            # skip files longer than 2 seconds
            try:
                info = torchaudio.info(filepath)
                duration = info.num_frames / info.sample_rate
            except Exception:
                # fall back to loading if info() not available
                tmp_y, tmp_sr = torchaudio.load(filepath, normalize=True)
                duration = tmp_y.shape[1] / tmp_sr

            if duration > self.input_time:
                continue

            y, sr, num_repetitions = self.create_augmented_wav(filepath, self.output_time, self.max_repetitions)
            y = self.add_musan_noise(y, sr, snr_db_range=(0,10), debug_wav=False)
            base_name = os.path.splitext(file)[0]
            npy_path = os.path.join(output_folder, base_name + "_spec_" + str(num_repetitions) + ".npy")
            self.create_spectrogram_npy(y, sr, npy_path)

if __name__ == '__main__':
    converter = DataConverter()
    y, sr, num_repetitions = converter.create_augmented_wav("drop.wav", converter.output_time, converter.max_repetitions)
    y = converter.add_musan_noise(y, sr, snr_db_range=(0,10), debug_wav=True)
    converter.create_spectrogram_npy(y, sr, "drop_spec_" + str(num_repetitions) + ".npy")
    converter.visualize_npy("drop_spec_" + str(num_repetitions) + ".npy", "drop_spec_viz.png")