import streamlit as st
import joblib
import pandas as pd
import re
import string
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import boto3
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt


# Initialize NLTK resources
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)  # Remove content within square brackets
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r"\n", " ", text)  # Remove newline characters
    text = re.sub(r"\w*\d\w*", "", text)  # Remove words with numbers
    text = re.sub(
        r"\s+", " ", text
    ).strip()  # Remove extra spaces and strip leading/trailing spaces
    text = " ".join(
        word for word in text.split() if word not in stop_words
    )  # Remove stopwords
    text = " ".join(stemmer.stem(word) for word in text.split())  # Apply stemming
    return text


# boto3 client for S3 (LocalStack)
s3 = boto3.client(
    "s3",
    endpoint_url="http://localstack:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)

bucket_name = "my-bucket"
model_file = "s3_model.pkl"


# Download model from S3
def download_model_from_s3():
    with open(model_file, "wb") as f:
        s3.download_fileobj(bucket_name, model_file, f)
        return joblib.load(model_file)


# Load local model
def load_local_model():
    loaded_pkl = joblib.load("best_model.pkl")
    model = loaded_pkl["model"]
    label = loaded_pkl["labels"]
    return model, label


# load model from S3
def load_s3_model():
    s3_model = download_model_from_s3()
    model = s3_model["model"]
    label = s3_model["labels"]
    return model, label


def monitor_score(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    return accuracy, precision, recall, f1


# Function to plot the performance metrics using matplotlib
def plot_metrics(accuracy, precision, recall, f1):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [accuracy, precision, recall, f1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(metrics, values, color=["blue", "green", "orange", "red"])
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Metrics", fontsize=14)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Metrics", fontsize=12)

    # Display the plot in Streamlit
    st.pyplot(fig)


st.title("Model Drift Monitoring")

model_source = st.selectbox("Select model source", ["Local", "S3"])

# Load model based on selection
if model_source == "S3":
    model, label_encoder = load_s3_model()
    st.write("Model successfully loaded from S3.")
else:
    st.write(f"Error loading model from S3, loading local model.")
    model, label_encoder = load_local_model()

user_input = st.text_input("Enter your message:")
if st.button("Predict"):
    if user_input:
        cleaned_text = clean_text(user_input)

        # Make the prediction
        prediction = model.predict([cleaned_text])
        predicted_label = label_encoder.inverse_transform(prediction)
        st.write(f"Prediction: {predicted_label[0]}")

        # Read data
        xdata = pd.read_csv("../artifact/transformer/xtest.csv")
        ytest = pd.read_csv("../artifact/transformer/ytest.csv")

        test_predictions = model.predict([clean_text(text) for text in xdata["clean"]])

        accuracy, precision, recall, f1 = monitor_score(
            ytest["status_encoded"], test_predictions
        )

        # Display performance metrics
        st.write(f"**Current Accuracy**: {accuracy:.4f}")
        st.write(f"**Current Precision**: {precision:.4f}")
        st.write(f"**Current Recall**: {recall:.4f}")
        st.write(f"**Current F1**: {f1:.4f}")

        # Plot performance metrics
        plot_metrics(accuracy, precision, recall, f1)

        report = classification_report(
            ytest["status_encoded"], test_predictions, output_dict=True
        )
        st.write("**Classification Report**:")
        st.dataframe(report)

        # Alert if model drift is detected (accuracy drops below threshold)
        if accuracy < 0.75:
            st.warning(
                "Accuracy dropped below threshold. Please consider retraining the model."
            )
    else:
        st.write("Please enter text to make a prediction.")
