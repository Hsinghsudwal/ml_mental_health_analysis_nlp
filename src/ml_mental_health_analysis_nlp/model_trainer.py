import pandas as pd
from src.constants import ARTIFACT, MODEL, BEST_MODEL, MONITOR, DEPLOYMENT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import joblib

import os


class ModelTrainer:
    def __init__(self):
        pass

    def modelTrainer(self, xtrain, ytrain, encoder):
        try:

            # Create the model pipeline
            clf = Pipeline(
                [
                    (
                        "vectorizer",
                        TfidfVectorizer(max_features=5000, stop_words="english"),
                    ),
                    ("classifier", LGBMClassifier()),
                ]
            )

            # Train the model
            clf.fit(xtrain, ytrain)

            # Create directories for model, monitor, and deployment paths
            model_path = os.path.join(ARTIFACT, MODEL)
            monitor_path = os.path.join(os.getcwd(), MONITOR)
            deployment_path = os.path.join(os.getcwd(), DEPLOYMENT)

            # Ensure all the necessary directories exist
            os.makedirs(model_path, exist_ok=True)
            os.makedirs(monitor_path, exist_ok=True)
            os.makedirs(deployment_path, exist_ok=True)

            # Save the model and encoder using joblib
            model_filename = os.path.join(model_path, BEST_MODEL)
            joblib.dump({"model": clf, "labels": encoder}, model_filename)

            # Save the model to the monitor and deployment directories as well
            monitor_model_filename = os.path.join(monitor_path, BEST_MODEL)
            deployment_model_filename = os.path.join(deployment_path, BEST_MODEL)

            joblib.dump({"model": clf, "labels": encoder}, monitor_model_filename)
            joblib.dump({"model": clf, "labels": encoder}, deployment_model_filename)

            print("Model Trainer Completed")
            return clf

        except Exception as e:
            raise Exception(e)
