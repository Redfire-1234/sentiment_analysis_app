import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

st.title("Sentiment Analysis for Goboult & Flipflop")

# Input from user
dataset = st.selectbox("Select Dataset", ["Goboult", "Flipflop"])
review = st.text_area("Enter your review here:")

# Load models and TF-IDF vectorizers
@st.cache_resource(show_spinner=False)
def load_models():
    with open('rf_goboult_model.pkl', 'rb') as f:
        goboult_model = pickle.load(f)
    with open('tfidf_goboult.pkl', 'rb') as f:
        goboult_tfidf = pickle.load(f)

    with open('rf_flipflop_model.pkl', 'rb') as f:
        flipflop_model = pickle.load(f)
    with open('tfidf_flipflop.pkl', 'rb') as f:
        flipflop_tfidf = pickle.load(f)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    return goboult_model, goboult_tfidf, flipflop_model, flipflop_tfidf, stop_words, lemmatizer

goboult_model, goboult_tfidf, flipflop_model, flipflop_tfidf, stop_words, lemmatizer = load_models()

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Prediction
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

