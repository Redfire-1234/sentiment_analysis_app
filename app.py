import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from huggingface_hub import hf_hub_download

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- Load models from Hugging Face Hub ---
# Replace with your actual HF model repo
HF_MODEL_REPO = "YourUsername/goboult-flipflop-sentiment-model"

goboult_model_file = hf_hub_download(HF_MODEL_REPO, "rf_goboult_model.pkl")
goboult_tfidf_file = hf_hub_download(HF_MODEL_REPO, "tfidf_goboult.pkl")
flipflop_model_file = hf_hub_download(HF_MODEL_REPO, "rf_flipflop_model.pkl")
flipflop_tfidf_file = hf_hub_download(HF_MODEL_REPO, "tfidf_flipflop.pkl")

with open(goboult_model_file, 'rb') as f:
    goboult_model = pickle.load(f)
with open(goboult_tfidf_file, 'rb') as f:
    goboult_tfidf = pickle.load(f)

with open(flipflop_model_file, 'rb') as f:
    flipflop_model = pickle.load(f)
with open(flipflop_tfidf_file, 'rb') as f:
    flipflop_tfidf = pickle.load(f)

# --- Streamlit UI ---
st.title("Sentiment Analysis for Goboult & Flipflop")

dataset = st.selectbox("Select Dataset", ["Goboult", "Flipflop"])
review = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review!")
    else:
        cleaned = preprocess_text(review)
        if dataset.lower() == "goboult":
            vectorized = goboult_tfidf.transform([cleaned])
            pred = goboult_model.predict(vectorized)[0]
        else:
            vectorized = flipflop_tfidf.transform([cleaned])
            pred = flipflop_model.predict(vectorized)[0]
        st.success(f"Predicted Sentiment: {pred}")


