import pytest
import random
import numpy as np
import pandas as pd

from pytest import approx


@pytest.fixture
def metadata():
    metadata_path = "D:\\shared/birdsong-recognition/train.csv"
    metadata = load_metadata(metadata_path)
    return metadata


def load_metadata(metadata_path):
    cols = ["ebird_code", "filename", "file_type", "sampling_rate", "duration"]
    metadata = pd.read_csv(metadata_path, usecols=cols)
    return metadata


def test_load_metadata(metadata):
    assert len(metadata.columns) == 5


# this is a bit slow
def split_dataset(metadata, train_f=0.85):
    train_metadata = pd.DataFrame(columns=metadata.columns)
    val_metadata = pd.DataFrame(columns=metadata.columns)

    metadata.set_index(["ebird_code", "filename"], drop=False, inplace=True)
    # unique birds
    keys = np.unique(tuple(m[0] for m in metadata.index))

    for k in keys:
        idx = list(filter(lambda x: x[0] == k, metadata.index))

        for i in idx:
            row = {col: metadata.loc[i, col] for col in metadata.columns}
            if random.random() < train_f:
                train_metadata = train_metadata.append(row, ignore_index=True)
            else:
                val_metadata = val_metadata.append(row, ignore_index=True)

    return train_metadata, val_metadata


def test_split_dataset(metadata):
    dataset_len = len(metadata)
    train_metadata, _ = split_dataset(metadata)
    assert len(train_metadata) / dataset_len == approx(0.85, abs=0.01)
