import os
import random

import torch
import torchaudio
from scipy import signal
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import random

random.seed(0)

class DataConverter:

    """
    
    Class for converting audio files into augmented spectrogram samples for training/testing, among other uses (e.g. histogram equalisation)

    """

    def __init__(self, input_time=1.0, output_time=10, max_repetitions=8, resample_rate=16000, musan_options=["noise"]):
        """init method"""

        self.input_time = input_time # maximum input audio length in seconds
        self.output_time = output_time
        self.max_repetitions = max_repetitions
        self.resample_rate = resample_rate

        search_dirs = []
        # Pre-scan MUSAN noise directories
        if "noise" in musan_options:
            folder = "/scratch/local/hdd/hani/musan/noise/"
            search_dirs += [os.path.join(folder, 'sound-bible'), os.path.join(folder, 'free-sound')]
        if "music" in musan_options:
            folder = "/scratch/local/hdd/hani/musan/music/"
            search_dirs += [os.path.join(folder, 'fma'), os.path.join(folder, 'fma-western-art'), os.path.join(folder, 'hd-classical'), os.path.join(folder, 'jamendo'), os.path.join(folder, 'rfm')]
        if "speech" in musan_options:
            folder = "/scratch/local/hdd/hani/musan/speech/"
            search_dirs += [os.path.join(folder, 'librivox'), os.path.join(folder, 'us-gov')]


        wav_files = []
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith('.wav'):
                        wav_files.append(os.path.join(root, f))
        self.musan_wav_files = wav_files
        if not self.musan_wav_files:
            print(f"Warning: no MUSAN .wav files found in {search_dirs}")

    def create_augmented_wav(self, file, output_time, max_repetitions, forced_repetitions=None, pause_between_reps = True, looped_wav = False, forced_time_window=None):
        """Create an augmented waveform from an input file."""
        #ignore non-wav files
        if not file.endswith(".wav"):
            return None, None, 0

        if looped_wav:
            y, sample_rate = self._extract_peak(file, max_time_window=self.input_time//2, forced_time_window=forced_time_window)
            y = torch.cat([y, torch.flip(y, dims=[1])], dim=1)
        else:
            y, sample_rate = self._extract_peak(file, max_time_window=self.input_time, forced_time_window=forced_time_window)
        #resample
        y = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.resample_rate)(y)
        sample_rate = self.resample_rate

        nonzero_mask = y.abs().sum(dim=0) > 0
        if nonzero_mask.any():
            y = y[:, nonzero_mask.nonzero()[0].item() : nonzero_mask.nonzero()[-1].item() + 1]


        file_samples = int(y.shape[1])
        target_samples = int(output_time * sample_rate)

        if file_samples > target_samples:
            print(file + " is too long")
            return None, None, 0

        #Decide rep number and resulting zeroes length to total desired output time
        max_repetitions = min(round(target_samples//file_samples), max_repetitions)

        if forced_repetitions is not None:
            num_repetitions = forced_repetitions
        else:
            num_repetitions = random.randint(0,max_repetitions)

        zeroes_samples = int(target_samples - num_repetitions * file_samples)

        #Randomly split zeroes lengths around repetitions
        # cuts = sorted([random.randint(0, zeroes_samples) for _ in range(num_repetitions)])
        # points = [0] + cuts + [zeroes_samples]
        # zeroes_lengths = []
        # for j in range(num_repetitions):
        #     zeroes_lengths.append(points[j+1]-points[j])

        if num_repetitions > 0:
            # Random start and end padding

            #gaussian method
            # start_padding = int(min(max(0, int(random.gauss(zeroes_samples*0.25, zeroes_samples*0.09))), int(zeroes_samples*0.5)))
            # end_padding = int(min(max(0, int(random.gauss(zeroes_samples*0.25, zeroes_samples*0.09))), int(zeroes_samples*0.5)))
            
            #uniform method
            # remaining_zeroes = int(random.randint(0,zeroes_samples))
            # start_padding = int(random.randint(0, zeroes_samples - remaining_zeroes))
            # end_padding = zeroes_samples - start_padding - remaining_zeroes

            #cut method
            cut1 = random.randint(0, zeroes_samples)
            cut2 = random.randint(0, zeroes_samples)
            start_padding = min(cut1, cut2)
            end_padding = zeroes_samples - max(cut1, cut2)
            remaining_zeroes = zeroes_samples - start_padding - end_padding

            if num_repetitions == 1 or not pause_between_reps:
                zeroes_lengths = [start_padding] + [0] * (num_repetitions - 1)
                end_padding += remaining_zeroes  # all leftover goes after the single repetition
            else:
                # Distribute remaining_zeroes approximately evenly between repetitions
                zeroes_lengths = [start_padding]

                weighted_zeroes = [max(0.01, random.gauss(1.0, 0.15)) for _ in range(num_repetitions - 1)]
                total_weight = sum(weighted_zeroes)
                weighted_zeroes = [w / total_weight for w in weighted_zeroes]

                gap_lengths = [int(w * remaining_zeroes) for w in weighted_zeroes]

                selected = 0
                while sum(gap_lengths) < remaining_zeroes:
                    gap_lengths[selected] += 1
                    selected += 1
                    if selected >= len(gap_lengths):
                        selected = 0
                while sum(gap_lengths) > remaining_zeroes:
                    gap_lengths[selected] -= 1
                    selected += 1
                    if selected >= len(gap_lengths):
                        selected = 0

                zeroes_lengths += gap_lengths

                # base_gap = remaining_zeroes // (num_repetitions - 1)
                # extra = remaining_zeroes % (num_repetitions - 1)

                # for i in range(num_repetitions - 1):
                #     gap = base_gap
                #     if i < extra:
                #         gap += 1  # distribute leftover
                #     # small random perturbation around gap
                #     perturb = int(random.gauss(0, remaining_zeroes*0.05))
                #     perturb = max(-gap, min(perturb, remaining_zeroes))  # keep valid
                #     zeroes_lengths.append(gap + perturb)
                
                # # Adjust zeroes_lengths to sum exactly remaining_zeroes
                # diff = remaining_zeroes - sum(zeroes_lengths)
                # end_padding += diff 
                # if end_padding < 0:
                #     zeroes_lengths[0] += end_padding
                # if zeroes_lengths[0] < 0:
                #     zeroes_lengths[-1] += zeroes_lengths[0]
                #     zeroes_lengths[0] = 0
                # #else:
                # #    god help you
                    
        else:
            start_padding = 0
            end_padding = zeroes_samples
            zeroes_lengths = []

        # ensure all zero lengths are ints
        zeroes_lengths = [int(x) for x in zeroes_lengths]
        start_padding = int(start_padding)
        end_padding = int(end_padding)

        output = torch.zeros(y.shape[0], target_samples)
        ptr = 0
        events = []
        for i in range(num_repetitions):
            ptr += zeroes_lengths[i]

            #track event timings
            start_time = round(ptr / sample_rate, 4)
            end_time = round((ptr + file_samples) / sample_rate, 4)
            events.append({"start": start_time, "end": end_time})

            output[:, ptr:ptr+file_samples] = y
            ptr += file_samples

        return output, sample_rate, num_repetitions, events

    def add_musan_noise(self, y, sr, snr_db_range=(23,30), debug_wav=False, forced_noise_file=None):
        """Add MUSAN noise to waveform y."""
        
        if forced_noise_file is not None:
            noise_file = forced_noise_file
        else:
            noise_file = random.choice(self.musan_wav_files)
        noise_y, noise_sr = torchaudio.load(noise_file, normalize=True)

        if noise_sr != sr:
            resampler = torchaudio.transforms.Resample(noise_sr, sr)
            noise_y = resampler(noise_y)
            noise_sr = sr

        target_len = y.shape[1]
        n_channels_target = y.shape[0]
        noise_len = noise_y.shape[1]

        orig_noise_segment = None

        if noise_len < target_len:
            noise_stream = torch.zeros(n_channels_target, target_len)
            insert_start = random.randint(0, target_len - noise_len)
            orig_noise_segment = noise_y.clone()
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

        if noise_y.shape[0] != n_channels_target:
            if noise_y.shape[0] == 1 and n_channels_target > 1:
                noise_y = noise_y.repeat(n_channels_target, 1)
            else:
                mono = noise_y.mean(dim=0, keepdim=True)
                noise_y = mono.repeat(n_channels_target, 1)

        snr_db = random.uniform(snr_db_range[0], snr_db_range[1])

        activity = y.abs().mean(dim=0)
        silence_thresh = 1e-4
        active_mask = activity > silence_thresh
        if active_mask.any():
            sig_rms = torch.sqrt(torch.mean(y[:, active_mask] ** 2))
        else:
            sig_rms = torch.sqrt(torch.mean(y ** 2))

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
            return y

        desired_noise_rms = sig_rms / (10 ** (snr_db / 20.0))
        scale = desired_noise_rms / (noise_rms + 1e-9)
        noise_y = noise_y * scale

        mixed = y + noise_y

        if debug_wav:
            torchaudio.save("debug_original.wav", y, sr)
            torchaudio.save("debug_noise.wav", noise_y, sr)
            torchaudio.save("debug_mixed.wav", mixed, sr)

        return mixed

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

    def _extract_peak(self, path, max_time_window=1.0, forced_time_window=None):
        """Extract peak volume moment from audio file"""

        x, sr = torchaudio.load(path)

        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)

        N = x.shape[1]

        if forced_time_window is not None:
            time_window = forced_time_window
        else:
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

    def batch_create_audio_dataset(self, input_folder, output_folder, use_per_file=2, add_noise=True, snr_db_range=(21, 25)):

        metadata = {}
        all_files = os.listdir(input_folder)
        total_wavs = sum(1 for f in all_files if f.lower().endswith('.wav'))
        counter = 0
        step = max(1, total_wavs // 500) if total_wavs > 0 else 1
        idx = 1

        for file in all_files:
            if not file.endswith(".wav"):
                continue
            counter += 1
            if counter % step == 0 or counter == total_wavs:
                print(f"Processed {counter}/{total_wavs} files...")

            filepath = os.path.join(input_folder, file)

            for j in range(0, use_per_file):
                y, sr, num_repetitions, events = self.create_augmented_wav(filepath, self.output_time, self.max_repetitions)
                if add_noise:
                    y = self.add_musan_noise(y, sr, snr_db_range=snr_db_range, debug_wav=False)
                base_name = os.path.splitext(file)[0]
                out_wav_path = os.path.join(output_folder, f"{idx:06d}.wav")
                torchaudio.save(out_wav_path, y, sr)

                #save metadata
                metadata[f"{idx:06d}"] = {
                    "original_source": file,
                    "num_repetitions": num_repetitions,
                    "events": events
                }
                idx += 1
        
        with open(os.path.join(output_folder, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

