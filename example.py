import librosa
import os
import pandas as pd
import numpy as np

from utils.preprocessor import Preprocessor
from utils.metadata_loader import MetadataLoader

DATA_PATH = "D:\\shared/birdsong-recognition"

if __name__=="__main__":
    metadata_loader = MetadataLoader(
        data_path=DATA_PATH
    )

    preprocessor = Preprocessor(metadata_loader)
    key = preprocessor.train_metadata.loc[0, "ebird_code"]
    filename = preprocessor.train_metadata.loc[0, "filename"]
    sampling_rate = preprocessor.train_metadata.loc[0, "sampling_rate"]
    duration = preprocessor.train_metadata.loc[0, "duration"]

    print("example audio file")
    print("bird: ", key)
    print("filename: ", filename)
    print("sampling rate: ", sampling_rate)
    print("duration: ", duration)

    stream = preprocessor.create_stream(
        key=key,
        filename=filename,
        sampling_rate=sampling_rate,
        duration=duration,
    )

    print("printing block sizes...")
    i = 0
    for s in stream:
        print(f"block {i}: ", len(s))
        i += 1