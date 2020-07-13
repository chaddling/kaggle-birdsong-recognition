import os
import glob
import librosa

from librosa.feature import melspectrogram

class Preprocessor:
    def __init__(self, metadata_loader):
        self.train_metadata = metadata_loader.train_metadata
        self.val_metadata = metadata_loader.val_metadata
        self.data_path = metadata_loader.data_path

    # we can just read the sampling rate from metadata so don't really need this
    def get_sampling_rate(self, file_location):
        file_path = os.path.join(self.data_path, file_location)
        return librosa.get_samplerate(file_path)

    def create_stream(self, key, filename, sampling_rate, duration, block_duration=5):
        file_path = os.path.join(self.data_path, f"train_audio/{key}/{filename}.wav")
        assert os.path.exists(file_path), "The audio file you are trying to split does not exist in .wav format."

        # conversion
        block_length = 128
        samples_per_block = sampling_rate * block_duration

        frame_length = samples_per_block // block_length
        hop_length = frame_length // 5

        stream = librosa.stream(
            path=file_path,
            block_length=block_length, # num. of frames per block
            frame_length=frame_length, # num. of samples per frame
            hop_length=hop_length, # num. of samples to advance between frames
            fill_value=0
        )
        return stream

    def compute_features(self, stream, sampling_rate):
        transformed_block = []
        for s in stream:
            transformed = melspectrogram(y=s, sampling_rate=sampling_rate)
            transformed_block.append(transformed)
        return transformed_block