"""
create_spec_dataset.py

Use DataConverter to create spectrogram dataset from audio files

"""

import os
import random
import data_converter

#create spectrogram dataset from audio files in a folder
converter = data_converter.DataConverter(musan_options=["noise", "speech", "music"])
converter.batch_create_spectrogram_samples(
    input_folder="/scratch/local/ssd/hani/FSD50K/FSD50K.eval_audio/",
    output_folder="/scratch/local/ssd/hani/RepeatVeryNoisy/test/",
    hist_eq='global',
    add_noise=True,
    use_per_file=2,
    snr_db_range=(0, 10),
)
