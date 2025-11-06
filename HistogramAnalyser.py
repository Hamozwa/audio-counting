import os
import wave
import matplotlib.pyplot as plt

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
    plt.savefig("output.png", dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def audio_count_histogram(folder):
    audio_counts = []

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            reps = file.split("_")[-1].split(".")[0]
            audio_counts.append(int(reps))

    plt.hist(audio_counts, bins=range(min(audio_counts), max(audio_counts) + 2), edgecolor='black', align='left')
    plt.xlabel("Audio Repetitions in File")
    plt.ylabel("Count")
    plt.title("Histogram of Audio Repetitions")
    plt.show()

# dir = "DataConverter_input"
# files = file_length_histogram(dir)

dir = "/scratch/local/ssd/hani/musan/noise/free-sound"
dir = "/scratch/local/ssd/hani/musan/noise/sound-bible"
file_length_histogram(dir)