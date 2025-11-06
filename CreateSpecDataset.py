import DataConverter

converter = DataConverter.DataConverter()
converter.batch_create_spectrogram_samples(
    input_folder="/scratch/local/ssd/hani/FSD50K/FSD50K.eval_audio/",
    output_folder="/scratch/local/ssd/hani/spec/test/"
)