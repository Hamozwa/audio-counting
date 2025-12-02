"""
create_spec_dataset.py

Use DataConverter to create spectrogram dataset from audio files

"""

import os
import random
import data_converter

#create spectrogram dataset from audio files in a folder
converter = data_converter.DataConverter()
converter.batch_create_spectrogram_samples(
    input_folder="/scratch/local/ssd/hani/FSD50K/FSD50K.eval_audio/",
    output_folder="/scratch/local/ssd/hani/RepeatSound/test/",
    hist_eq='global',
    add_noise=False,
    use_per_file=2,
)

#random visualization of dataset sample

# converter = data_converter.DataConverter()
# choice = random.choice(os.listdir("/scratch/local/ssd/hani/countix-av-spec/train/"))
# print("Visualizing:", choice)
# converter.visualize_npy(os.path.join("/scratch/local/ssd/hani/countix-av-spec/train/", choice), "random_spec.png")