import random
import os
import numpy as np
import pandas as pd


class MetadataLoader:
    def __init__(self, data_path, train_f=0.85):
        self.train_f = train_f
        self.data_path = data_path

        dataset_metadata = self._load_metadata(os.path.join(data_path, "train.csv"))
        self._train_metadata, self._val_metadata = self._split_dataset(dataset_metadata)

    @property
    def train_metadata(self):
        return self._train_metadata

    @property
    def val_metadata(self):
        return self._val_metadata

    def _load_metadata(self, metadata_path):
        # only load the "useful" metadata columns...
        cols = ["ebird_code", "filename", "file_type", "sampling_rate", "duration"]
        metadata = pd.read_csv(metadata_path, usecols=cols)
        
        metadata.filename = metadata.filename.apply(lambda x: x.split(".")[0])
        metadata.sampling_rate = metadata.sampling_rate.apply(lambda x: int(x.split()[0]))
        return metadata

    def _split_dataset(self, metadata):
        train_metadata = pd.DataFrame(columns=metadata.columns)
        val_metadata = pd.DataFrame(columns=metadata.columns)

        metadata.set_index(["ebird_code", "filename"], drop=False, inplace=True)
        # unique birds
        keys = np.unique(tuple(m[0] for m in metadata.index))

        for k in keys:
            idx = list(filter(lambda x: x[0] == k, metadata.index))

            for i in idx:
                row = {col: metadata.loc[i, col] for col in metadata.columns}
                if random.random() < self.train_f:
                    train_metadata = train_metadata.append(row, ignore_index=True)
                else:
                    val_metadata = val_metadata.append(row, ignore_index=True)
        return train_metadata, val_metadata
