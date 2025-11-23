
# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy files
COPY app.py .
COPY rf_goboult_model.pkl .
COPY tfidf_goboult.pkl .
COPY rf_flipflop_model.pkl .
COPY tfidf_flipflop.pkl .

# Install dependencies
RUN pip install --no-cache-dir flask scikit-learn nltk pandas

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
