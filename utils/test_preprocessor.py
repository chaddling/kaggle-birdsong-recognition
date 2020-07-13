import librosa
import pytest
import os
import subprocess

from librosa.feature import melspectrogram
from utils.preprocessor import Preprocessor
from utils.metadata_loader import MetadataLoader


def load_audio_file():
    y, sr = librosa.load("D:\\shared/birdsong-recognition/train_audio/aldfly/XC134874.mp3", sr=None)
    return y, sr

def test_load_audio_file():
    y, sr = load_audio_file()
    assert len(y)
    assert(sr)


@pytest.fixture
def preprocessor():
    metadata_loader = MetadataLoader("D:\\shared/birdsong-recognition")
    return Preprocessor(metadata_loader)


def test_preprocessor(preprocessor):
    assert preprocessor


def test_preprocessor_get_sampling_rate(preprocessor):
    key = preprocessor.train_metadata.loc[0, 'ebird_code']
    filename = preprocessor.train_metadata.loc[0, 'filename']
    file_type = preprocessor.train_metadata.loc[0, 'file_type']
    file_location = f"train_audio/{key}/{filename}.{file_type}"

    sampling_rate =preprocessor.train_metadata.loc[0, 'sampling_rate'] # a string e.g. `44100 (Hz)`
    sampling_rate = int(sampling_rate.split()[0])

    file_path = os.path.join(preprocessor.data_path, file_location)
    sr = preprocessor.get_sampling_rate(file_path)
    assert sr == sampling_rate


def test_preprocessor_get_stream(preprocessor):
    key = preprocessor.train_metadata.loc[0, 'ebird_code']
    filename = preprocessor.train_metadata.loc[0, 'filename']
    file_type = preprocessor.train_metadata.loc[0, 'file_type']
    duration = preprocessor.train_metadata.loc[0, 'duration']

    sampling_rate = preprocessor.train_metadata.loc[0, 'sampling_rate']
    
    output_location = f"train_audio/{key}/{filename}.wav"
    output_path = os.path.join(preprocessor.data_path, output_location)
    if not os.path.exists(output_path):
        input_location = f"train_audio/{key}/{filename}.{file_type}"
        input_path = os.path.join(preprocessor.data_path, input_location)
        subprocess.check_output(
           f"ffmpeg -i {input_path} -acodec pcm_s16le -ac 1 -ar {sampling_rate} {output_path}",
           shell=True,
        )

    stream = preprocessor.create_stream(key, filename, sampling_rate, duration)
    assert len(list(stream)) == 20 # how to calculate this?
