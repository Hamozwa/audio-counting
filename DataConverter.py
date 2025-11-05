import torchaudio
import torch
import random
import os

input_dir = "DataConverter_input"
output_dir = "DataConverter_output"

output_time = 16
max_repetitions = 8 #lowered if clips cannot fit into this time

for file in os.listdir(input_dir):
    #ignore non-wav files
    if not file.endswith(".wav"):
        continue

    y, sample_rate = torchaudio.load(os.path.join(input_dir, file),normalize=True)
    file_samples = y.shape[1]
    target_samples = output_time * sample_rate

    if file_samples > target_samples:
        print(file + " is too long")
        continue

    #Decide rep number and resulting zeroes length to total desired output time
    max_repetitions = min(round(target_samples//file_samples), max_repetitions)
    num_repetitions = random.randint(0,max_repetitions)
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

    #rename file and save in output folder
    root, ext = os.path.splitext(file)
    new_name = f"{root}_{num_repetitions}{ext}"
    torchaudio.save(os.path.join(output_dir, new_name), output, sample_rate)