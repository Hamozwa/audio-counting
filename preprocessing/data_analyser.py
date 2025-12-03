"""
data_analyser.py

Functions to analyse audio dataset files

"""

import os
import wave
import matplotlib.pyplot as plt
import random
import data_converter
import shutil

def file_length_histogram(folder):
    durations = []

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            with wave.open(os.path.join(folder, file), 'r') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                durations.append(frames / float(rate))

    plt.hist(durations, bins=50, edgecolor='black')
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of files")
    plt.title("WAV File Duration Histogram")
    plt.show()

def audio_count_histogram(folder):
    audio_counts = []

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            reps = file.split("_")[-1].split(".")[0]
            audio_counts.append(int(reps))

    plt.hist(audio_counts, bins=range(min(audio_counts), max(audio_counts) + 2), edgecolor='black', align='left')
    plt.xlabel("Audio Repetitions in File")
    plt.ylabel("Count")
    plt.title("Histogram of Audio Repetitions")
    plt.savefig("audio_repetitions_histogram.png", dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualise_random_sample(path="/scratch/local/ssd/hani/countix-av-spec/train/"):
    converter = data_converter.DataConverter()
    choice = random.choice(os.listdir(path))
    shutil.copy(os.path.join(path, choice), "random_spec.npy")
    print("Visualizing:", choice)
    converter.visualize_npy(os.path.join(path, choice), "random_spec.png")


if __name__ == "__main__":
    # dir = "/scratch/local/ssd/hani/FSD50K/FSD50K.eval_audio/"
    # files = file_length_histogram(dir)

    # dir = "/scratch/local/ssd/hani/spec3/test/"
    # audio_count_histogram(dir)
    visualise_random_sample()