import pandas as pd
from src.constants import ARTIFACT, TRANSFORMATION, XTRAIN, XTEST, YTRAIN, YTEST
from src.utilities import helper_text
from sklearn.preprocessing import LabelEncoder
import os


class DataTransformation:
    def __init__(self):
        pass

    def dataTransformation(self, train, test):
        try:

            # get clean
            train["clean"] = train["statement"].apply(helper_text)
            test["clean"] = test["statement"].apply(helper_text)
            # print(train)

            encoder = LabelEncoder()
            train["status_encoded"] = encoder.fit_transform(train["status"])
            test["status_encoded"] = encoder.transform(test["status"])

            xtrain = train["clean"]
            xtest = test["clean"]
            ytrain = train["status_encoded"]
            ytest = test["status_encoded"]

            transformer_path = os.path.join(ARTIFACT, TRANSFORMATION)
            os.makedirs(transformer_path, exist_ok=True)

            xtrain.to_csv(os.path.join(transformer_path, str(XTRAIN)))
            xtest.to_csv(os.path.join(transformer_path, str(XTEST)))

            ytrain.to_csv(os.path.join(transformer_path, str(YTRAIN)))
            ytest.to_csv(os.path.join(transformer_path, str(YTEST)))

            print("Data Transformation Completed")
            return xtrain, xtest, ytrain, ytest, encoder

        except Exception as e:
            raise Exception(e)
