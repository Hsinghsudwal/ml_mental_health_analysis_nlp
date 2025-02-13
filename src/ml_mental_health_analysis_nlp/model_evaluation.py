import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    recall_score,
    f1_score,
    accuracy_score,
    precision_score,
    classification_report,
)
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from src.constants import ARTIFACT, EVALUATION, METRIC_JSON, CM_MATRIX, CLASS_REPORT


class ModelEvaluation:

    def __init__(self) -> None:
        pass

    def metric_score(y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        return accuracy, precision, recall, f1

    def modelEvaluate(self, model, xtest, ytest):
        try:
            # Make predictions
            y_pred = model.predict(xtest)
            accuracy, precision, recall, f1 = ModelEvaluation.metric_score(
                ytest, y_pred
            )

            metrics_dict = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
            }

            # Print out the metrics
            print("Accuracy: ", str(round(accuracy, 2) * 100) + "%")
            print("Precision: ", round(precision, 2))
            print("Recall: ", round(recall, 2))
            print("F1: ", round(f1, 2))

            # Save the metrics in a JSON file
            evaluate_path = os.path.join(ARTIFACT, EVALUATION)
            os.makedirs(evaluate_path, exist_ok=True)
            evaluate_filename = os.path.join(evaluate_path, METRIC_JSON)
            with open(evaluate_filename, "w") as f:
                json.dump(
                    {
                        "metrics": metrics_dict,
                    },
                    f,
                    indent=4,
                )

            # Generate and save Confusion Matrix
            conf_matrix = confusion_matrix(ytest, y_pred)
            print("Confusion Matrix:")
            print(conf_matrix)
            sns.heatmap(conf_matrix, annot=True, fmt="g")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            cm_path = os.path.join(ARTIFACT, EVALUATION, CM_MATRIX)
            plt.savefig(cm_path, dpi=120)
            plt.close()

            # Save classification report as text
            class_report = classification_report(ytest, y_pred)
            print("Classification Report:")
            print(class_report)

            class_report_txt_path = os.path.join(ARTIFACT, EVALUATION, CLASS_REPORT)
            with open(class_report_txt_path, "w") as f:
                f.write(class_report)

            print("Model Evaluation Completed")

        except Exception as e:
            raise Exception(f"Error in model evaluation: {str(e)}")
