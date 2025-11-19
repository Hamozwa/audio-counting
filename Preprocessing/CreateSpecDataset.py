import DataConverter

#create spectrogram dataset from audio files in a folder
converter = DataConverter.DataConverter()
# converter.batch_create_spectrogram_samples(
#     input_folder="/scratch/local/ssd/hani/FSD50K/FSD50K.eval_audio/",
#     output_folder="/scratch/local/ssd/hani/spec/test/",
#     hist_eq='clahe'  # apply CLAHE histogram equalization during preprocessing
# )

#random visualization of dataset sample
import os
import random

choice = random.choice(os.listdir("/scratch/local/ssd/hani/spec/test/"))
print("Visualizing:", choice)
converter.visualize_npy(os.path.join("/scratch/local/ssd/hani/spec/test/", choice), "random_spec.png")