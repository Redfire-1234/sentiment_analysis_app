
from flask import Flask, request, jsonify
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import os

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data (will use cached if available)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

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

@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'message': 'Sentiment Analysis API for Goboult & Flipflop is running!',
        'endpoints': {
            '/': 'Health check',
            '/predict': 'POST - Predict sentiment'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        review = data.get('review', '').strip()
        if not review:
            return jsonify({'error': 'Review text is required'}), 400
            
        dataset = data.get('dataset', 'goboult').lower()

        cleaned = preprocess_text(review)
        
        if not cleaned:
            return jsonify({'error': 'Review text is too short'}), 400

        if dataset == 'goboult':
            vectorized = goboult_tfidf.transform([cleaned])
            pred = goboult_model.predict(vectorized)[0]
        elif dataset == 'flipflop':
            vectorized = flipflop_tfidf.transform([cleaned])
            pred = flipflop_model.predict(vectorized)[0]
        else:
            return jsonify({'error': 'Dataset must be "goboult" or "flipflop"'}), 400

        return jsonify({'review': review, 'dataset': dataset, 'predicted_sentiment': pred})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
