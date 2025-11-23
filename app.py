
from flask import Flask, request, jsonify
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load Goboult model & TF-IDF
with open('rf_goboult_model.pkl', 'rb') as f:
    goboult_model = pickle.load(f)
with open('tfidf_goboult.pkl', 'rb') as f:
    goboult_tfidf = pickle.load(f)

# Load Flipflop model & TF-IDF
with open('rf_flipflop_model.pkl', 'rb') as f:
    flipflop_model = pickle.load(f)
with open('tfidf_flipflop.pkl', 'rb') as f:
    flipflop_tfidf = pickle.load(f)

# Preprocessing setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return "Sentiment Analysis API for Goboult & Flipflop is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data.get('review', '')
    dataset = data.get('dataset', 'goboult').lower()  # Default to goboult

    cleaned = preprocess_text(review)

    if dataset == 'goboult':
        vectorized = goboult_tfidf.transform([cleaned])
        pred = goboult_model.predict(vectorized)[0]
    elif dataset == 'flipflop':
        vectorized = flipflop_tfidf.transform([cleaned])
        pred = flipflop_model.predict(vectorized)[0]
    else:
        return jsonify({'error': 'Dataset must be "goboult" or "flipflop"'}), 400

    return jsonify({'review': review, 'dataset': dataset, 'predicted_sentiment': pred})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
