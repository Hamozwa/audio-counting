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
    input_folder="/scratch/local/ssd/hani/FSD50K/train/",
    output_folder="/scratch/local/ssd/hani/RepeatSoundNoisy/train/",
    hist_eq='global',
    add_noise=True,
    use_per_file=2,
)
