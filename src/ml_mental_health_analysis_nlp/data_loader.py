import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.constants import ARTIFACT, TEST_SIZE, RAW, TRAIN_DATA, TEST_DATA
import os


class DataLoader:
    def __init__(self):
        pass

    def dataLoader(self, path):
        try:

            data = pd.read_csv(path, index_col=0)

            data.dropna(inplace=True)

            # split into train test data
            train_data, test_data = train_test_split(
                data, test_size=TEST_SIZE, random_state=42
            )

            raw_data_path = os.path.join(ARTIFACT, RAW)
            os.makedirs(raw_data_path, exist_ok=True)
            # save train test to csv
            train_data.to_csv(os.path.join(raw_data_path, str(TRAIN_DATA)), index=False)
            test_data.to_csv(os.path.join(raw_data_path, str(TEST_DATA)), index=False)
            print("Data Ingestion Completed")
            return train_data, test_data

        except Exception as e:
            raise Exception(e)
