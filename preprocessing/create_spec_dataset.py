"""
create_spec_dataset.py

Use DataConverter to create spectrogram dataset from audio files

"""

import os
import random
import preprocessing.data_converter as data_converter

#create spectrogram dataset from audio files in a folder
# converter = DataConverter.DataConverter()
# converter.batch_create_spectrogram_samples(
#     input_folder="/scratch/local/ssd/hani/FSD50K/FSD50K.eval_audio/",
#     output_folder="/scratch/local/ssd/hani/RepeatSoundNoisy/test/",
#     hist_eq='global',
#     add_noise=True,
#     use_per_file=8
# )

#random visualization of dataset sample

converter = data_converter.DataConverter()
choice = random.choice(os.listdir("/scratch/local/ssd/hani/countix-av-spec/train/"))
print("Visualizing:", choice)
converter.visualize_npy(os.path.join("/scratch/local/ssd/hani/countix-av-spec/train/", choice), "random_spec.png")