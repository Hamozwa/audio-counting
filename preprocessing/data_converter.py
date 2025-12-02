import os
import random

import torch
import torchaudio
from scipy import signal
import numpy as np
import cv2
import matplotlib.pyplot as plt

class DataConverter:

    """
    
    Class for converting audio files into augmented spectrogram samples for training/testing

    """

    def __init__(self, input_time=1.0, output_time=10, max_repetitions=8, resample_rate=16000):
        """init method"""

        self.input_time = input_time # maximum input audio length in seconds
        self.output_time = output_time
        self.max_repetitions = max_repetitions
        self.resample_rate = resample_rate

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

    def create_augmented_wav(self, file, output_time, max_repetitions, forced_repetitions=None):
        """Create an augmented waveform from an input file."""

        #ignore non-wav files
        if not file.endswith(".wav"):
            return None, None, 0

        y, sample_rate = self._extract_peak(file, max_time_window=self.input_time)

        #resample
        y = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.resample_rate)(y)
        sample_rate = self.resample_rate


        file_samples = y.shape[1]
        target_samples = output_time * sample_rate

        if file_samples > target_samples:
            print(file + " is too long")
            return None, None, 0

        #Decide rep number and resulting zeroes length to total desired output time
        max_repetitions = min(round(target_samples//file_samples), max_repetitions)

        if forced_repetitions is not None:
            num_repetitions = forced_repetitions
        else:
            num_repetitions = random.randint(0,max_repetitions)

        zeroes_samples = target_samples - num_repetitions*file_samples

        #Randomly split zeroes lengths around repetitions
        cuts = sorted([random.randint(0, zeroes_samples) for _ in range(num_repetitions)])
        points = [0] + cuts + [zeroes_samples]
        zeroes_lengths = []
        for j in range(num_repetitions):
            zeroes_lengths.append(points[j+1]-points[j])

        if num_repetitions > 0:
            # Random start and end padding
            start_padding = min(max(0, int(random.gauss(zeroes_samples*0.25, zeroes_samples*0.09))), zeroes_samples*0.5)
            end_padding = min(max(0, int(random.gauss(zeroes_samples*0.25, zeroes_samples*0.09))), zeroes_samples*0.5)

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

    def add_musan_noise(self, y, sr, snr_db_range=(23,30), debug_wav=False, forced_noise_file=None):
        """Add MUSAN noise to waveform y."""
        
        if forced_noise_file is not None:
            noise_file = forced_noise_file
        else:
            noise_file = random.choice(self.musan_wav_files)
        print(f"Adding noise from: {noise_file}")
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

        # Keep original (unpadded) noise segment for correct RMS calculation
        orig_noise_segment = None

        if noise_len < target_len:
            # Place the noise once at a random location inside the target length
            noise_stream = torch.zeros(n_channels_target, target_len)
            insert_start = random.randint(0, target_len - noise_len)
            # save unpadded original for RMS calc
            orig_noise_segment = noise_y.clone()
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
        print(f"Chosen SNR (dB): {snr_db}")

        # Compute signal RMS on active (non-silent) frames to avoid silent-padding bias
        activity = y.abs().mean(dim=0)
        silence_thresh = 1e-4
        active_mask = activity > silence_thresh
        if active_mask.any():
            sig_rms = torch.sqrt(torch.mean(y[:, active_mask] ** 2))
        else:
            sig_rms = torch.sqrt(torch.mean(y ** 2))

        # Compute noise RMS from the original unpadded noise segment if available
        if orig_noise_segment is not None:
            if orig_noise_segment.shape[0] == 1 and n_channels_target > 1:
                orig_noise_segment = orig_noise_segment.repeat(n_channels_target, 1)
            elif orig_noise_segment.shape[0] != n_channels_target:
                mono = orig_noise_segment.mean(dim=0, keepdim=True)
                orig_noise_segment = mono.repeat(n_channels_target, 1) 

            noise_rms = torch.sqrt(torch.mean(orig_noise_segment ** 2))
        else:
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

    def musan_noise_spectrogram(self, target_shape, sr, nperseg=512, noverlap=256, forced_noise_file=None):
        """ Return a MUSAN noise power-spectrogram shaped (freq_bins, time_frames) """
        num_bins, num_frames = target_shape
        hop = nperseg - noverlap
        required_samples = int((max(1, num_frames - 1) * hop) + nperseg)

        # pick noise file
        if forced_noise_file is not None:
            noise_file = forced_noise_file
        else:
            noise_file = random.choice(self.musan_wav_files)
        noise_y, noise_sr = torchaudio.load(noise_file, normalize=True)

        # resample if needed
        if noise_sr != sr:
            resampler = torchaudio.transforms.Resample(noise_sr, sr)
            noise_y = resampler(noise_y)
            noise_sr = sr

        # convert to mono waveform numpy
        noise_wave = noise_y.mean(dim=0).cpu().numpy()
        noise_len = noise_wave.shape[0]

        # if too short: repeat it until long enough (simple and deterministic)
        if noise_len < required_samples:
            reps = int(np.ceil(required_samples / float(noise_len)))
            noise_wave = np.tile(noise_wave, reps)[:required_samples]
        else:
            start = random.randint(0, noise_len - required_samples)
            noise_wave = noise_wave[start:start + required_samples]

        # compute power spectrogram with same STFT params
        _, _, Sxx = signal.spectrogram(noise_wave, sr, nperseg=nperseg, noverlap=noverlap)
        # Sxx is power; crop/pad time dimension to match num_frames
        if Sxx.shape[1] >= num_frames:
            Sxx = Sxx[:, :num_frames]
        else:
            # pad with small epsilon columns if unexpectedly short
            pad_cols = num_frames - Sxx.shape[1]
            Sxx = np.concatenate([Sxx, np.zeros((Sxx.shape[0], pad_cols), dtype=Sxx.dtype)], axis=1)

        # Ensure freq bins match — if not, crop or pad (rare if same nperseg used)
        if Sxx.shape[0] > num_bins:
            Sxx = Sxx[:num_bins, :]
        elif Sxx.shape[0] < num_bins:
            pad_rows = num_bins - Sxx.shape[0]
            Sxx = np.concatenate([Sxx, np.zeros((pad_rows, Sxx.shape[1]), dtype=Sxx.dtype)], axis=0)

        return Sxx  # power spectrogram (freq x time)

    def apply_histogram_equalisation(self, spectrogram, method='clahe', clipLimit=8.0, tileGridSize=(4,4)):
        if method is None:
            return spectrogram

        # preserve original min/max so we can map back after equalisation
        orig_min = float(np.min(spectrogram))
        orig_max = float(np.max(spectrogram))
        if orig_max - orig_min < 1e-9:
            return spectrogram

        # scale to 0-255
        scaled = (spectrogram - orig_min) / (orig_max - orig_min)
        scaled = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

        if method == 'global':
            eq = cv2.equalizeHist(scaled)
        elif method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            eq = clahe.apply(scaled)
        else:
            return spectrogram

        # map back to original float range
        eq_f = eq.astype(np.float32) / 255.0
        eq_f = eq_f * (orig_max - orig_min) + orig_min
        return eq_f

    def create_spectrogram_npy(self, y, sr, out_npy_path = None, nperseg=512, noverlap=256, hist_eq=None, add_noise=False):
        waveform = y.mean(dim=0).cpu().numpy()
        _, _, spectrogram = signal.spectrogram(waveform, sr, nperseg=nperseg, noverlap=noverlap)
        spectrogram = np.log(spectrogram + 1e-7)

        # apply histogram equalization
        if hist_eq is not None:
            spectrogram = self.apply_histogram_equalisation(spectrogram, method=hist_eq)

        if add_noise:
            # spectrogram currently holds log(power + eps)
            spec_power = np.exp(spectrogram) - 1e-7  # back to linear power

            # get a noise power-spectrogram with matching shape
            noise_power = self.musan_noise_spectrogram(spec_power.shape, sr, nperseg=nperseg, noverlap=noverlap, forced_noise_file=None)

            # choose SNR (dB) in same style as waveform-based method
            snr_db = random.uniform(23, 30)
            snr_linear = 10.0 ** (snr_db / 10.0)

            # compute mean powers and scale noise power to achieve desired SNR (power ratio)
            power_s = np.mean(spec_power) + 1e-12
            power_n = np.mean(noise_power) + 1e-12
            desired_noise_power = power_s / snr_linear
            scale = desired_noise_power / power_n

            # mix in power-domain
            spec_mixed = spec_power + noise_power * scale

            # convert back to log-power for the rest of the pipeline
            spectrogram = np.log(spec_mixed + 1e-7)

            measured_snr_db = 10*np.log10(np.mean(spec_power) / (np.mean((noise_power * scale)) + 1e-12))
            #(f"Added MUSAN spectrogram noise: target SNR {snr_db:.1f} dB, measured SNR {measured_snr_db:.2f} dB")

        # normalise to zero mean and unit variance
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        noise = np.random.normal(0, 0.05, spectrogram.shape)
        spectrogram = spectrogram + noise

        spectrogram = cv2.resize(spectrogram, (224, 224))

        # save spectrogram as float32
        if out_npy_path is not None:
            np.save(out_npy_path, spectrogram.astype(np.float32))
        else:
            return spectrogram

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

        plt.figure(figsize=(224/150, 224/150))  # width, height in inches = pixels / DPI
        plt.imshow(a, origin='lower', aspect='equal', cmap=cmap)
        plt.axis('off')
        plt.savefig(out_png, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()

    def batch_create_spectrogram_samples(self, input_folder, output_folder, add_noise=True, hist_eq=None, use_per_file = 1):
        """
        
        Batch process all wav files in input_folder, creating spectrogram .npy files in output_folder
        
        """


        # simple progress counter
        all_files = os.listdir(input_folder)
        total_wavs = sum(1 for f in all_files if f.lower().endswith('.wav'))
        counter = 0
        step = max(1, total_wavs // 500) if total_wavs > 0 else 1

        for file in all_files:
            if not file.endswith(".wav"):
                continue
            counter += 1
            if counter % step == 0 or counter == total_wavs:
                print(f"Processed {counter}/{total_wavs} files...")

            filepath = os.path.join(input_folder, file)

            for j in range(0, use_per_file):
                y, sr, num_repetitions = self.create_augmented_wav(filepath, self.output_time, self.max_repetitions)
                base_name = os.path.splitext(file)[0]
                npy_path = os.path.join(output_folder, base_name + "_spec_" + str(j) + "_" + str(num_repetitions) + ".npy")
                self.create_spectrogram_npy(y, sr, npy_path, hist_eq=hist_eq, add_noise=add_noise)

    def _extract_peak(self, path, max_time_window=1.0):
        """Extract peak volume moment from audio file"""

        x, sr = torchaudio.load(path)

        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)

        N = x.shape[1]

        time_window = random.uniform(0.1, max_time_window)
        window_len = int(sr * time_window)

        peak_idx = torch.argmax(torch.abs(x))
        peak_position = random.randint(0, window_len - 1)

        start = peak_idx - peak_position
        end = start + window_len

        #keep in audio bound
        if start < 0:
            start = 0
            end = window_len

        if end > N:
            end = N
            start = max(0, end - window_len)

        segment = x[:, start:end]

        # Pad if needed (file shorter than 1s or window clipped)
        if segment.shape[1] < window_len:
            pad_left = 0
            pad_right = 0
            if start == 0:
                pad_left = 0
                pad_right = window_len - segment.shape[1]
            elif end == N:
                pad_left = window_len - segment.shape[1]
                pad_right = 0
            segment = torch.nn.functional.pad(segment, (pad_left, pad_right))
        return segment, sr

if __name__ == '__main__':
    converter = DataConverter()
    
    choice = random.choice(os.listdir("/scratch/local/ssd/hani/FSD50K/train/"))
    filepath = os.path.join("/scratch/local/ssd/hani/FSD50K/train/", choice)

    y, sr, num_repetitions = converter.create_augmented_wav(filepath, converter.output_time, converter.max_repetitions)
    print("Repetitions:", num_repetitions)
    converter.create_spectrogram_npy(y, sr, "drop_spec_" + str(num_repetitions) + ".npy", hist_eq="global", add_noise=True)
    converter.visualize_npy("drop_spec_" + str(num_repetitions) + ".npy", "drop_spec_global.png")
