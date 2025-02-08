import pandas as pd

from src.ml_nlp_analysis.data_loader import DataLoader
from src.ml_nlp_analysis.data_transformation import DataTransformation
from src.ml_nlp_analysis.model_trainer import ModelTrainer
from src.ml_nlp_analysis.model_evaluation import ModelEvaluation


def training_pipeline():
    path = "data/data.csv"
    # path = "data/simple.csv"
    dataload = DataLoader()
    train, test = dataload.dataLoader(path)

    datatrans = DataTransformation()
    xtrain, xtest, ytrain, ytest, encoder = datatrans.dataTransformation(train, test)

    modeltrain = ModelTrainer()
    model = modeltrain.modelTrainer(xtrain, ytrain, encoder)

    modelevalu = ModelEvaluation()
    modelevalu.modelEvaluate(model, xtest, ytest)
