import streamlit as st
import joblib
import re
import string
import boto3
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


nltk.download("stopwords", quiet=True)


stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

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


# download model from S3
def download_model_from_s3():
    with open(model_file, "wb") as f:
        try:
            s3.download_fileobj(bucket_name, model_file, f)
        except Exception as e:
            print(f"Error downloading model from S3: {e}")
            raise
    return joblib.load(model_file)


# load local model
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


# clean input text
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


st.title("Chatbot Prediction")

model_source = st.selectbox("Select model source", ["Local", "S3"])

# Load model based on selection
if model_source == "S3":
    model, label = load_s3_model()
else:
    print("Local")
    model, label = load_local_model()


user_input = st.text_input("Enter your message:")


if st.button("Predict"):
    if user_input:
        cleaned_text = clean_text(user_input)

        # Make prediction
        prediction = model.predict([cleaned_text])
        predicted_label = label.inverse_transform(prediction)[0]
        st.write(f"Prediction: {predicted_label}")
    else:
        st.write("Please enter some text to get a prediction.")
