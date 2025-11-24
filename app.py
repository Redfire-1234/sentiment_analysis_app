import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
nltk.download('punkt_tab')  # Changed from 'punkt'
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Added for better lemmatization

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- Load models from local files in Space ---
@st.cache_resource
def load_models():
    with open('rf_goboult_model.pkl', 'rb') as f:
        goboult_model = pickle.load(f)
    with open('tfidf_goboult.pkl', 'rb') as f:
        goboult_tfidf = pickle.load(f)
    with open('rf_flipflop_model.pkl', 'rb') as f:
        flipflop_model = pickle.load(f)
    with open('tfidf_flipflop.pkl', 'rb') as f:
        flipflop_tfidf = pickle.load(f)
    return goboult_model, goboult_tfidf, flipflop_model, flipflop_tfidf

goboult_model, goboult_tfidf, flipflop_model, flipflop_tfidf = load_models()

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


