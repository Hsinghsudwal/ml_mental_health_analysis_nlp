import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

stemmer = nltk.SnowballStemmer("english")
stop_words = stopwords.words("english")
nltk.download("punkt")
nltk.download("wordnet")


def helper_text(text):
    text = str(text).lower()  # Convert to string and lowercase
    text = re.sub(r"\[.*?\]", "", text)  # Remove text in square brackets
    text = re.sub(r"https?://\S+|www\.\S+|\[.*?\]\(.*?\)", "", text)  # Remove URLs
    text = re.sub(r"<.*?>+", "", text)  # Remove HTML tags
    text = re.sub(r"@\w+", "", text)  # Remove @ mentions
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r"\n", " ", text)  # Remove newlines
    text = re.sub(r"\w*\d\w*", "", text)  # Remove words with numbers
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading and trailing spaces
    text = " ".join(word for word in text.split(" ") if word not in stop_words)
    text = " ".join(stemmer.stem(word) for word in text.split(" "))
    return text
