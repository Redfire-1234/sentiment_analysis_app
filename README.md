# Sentiment Analysis for Goboult & Flipflop


## References:  
- [Scikit-learn](https://scikit-learn.org/stable/)  
- [Streamlit](https://streamlit.io/)  
- [NLTK](https://www.nltk.org/)  
- [PyTorch](https://pytorch.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [Seaborn](https://seaborn.pydata.org/)  
- [WordCloud](https://github.com/amueller/word_cloud)  
- [ngrok](https://ngrok.com/)  

---

## Project Overview:  
This project performs **Sentiment Analysis** on reviews from Goboult and Flipflop products.  
The Streamlit application allows users to input any review text and view the predicted sentiment: **Positive** or **Negative**.  

---

## Dataset:  
- Goboult Reviews: `goboult_shadow_review.csv`  
- Flipflop Reviews: `flipflop_review.csv`  
- Labels:  
  - Positive → Rating >= 4  
  - Negative → Rating <= 2  
  - Neutral reviews are excluded  
- Dataset can be expanded for better performance  

---

## Preprocessing:  
- Text cleaning and normalization  
- Lowercasing, punctuation removal  
- Tokenization, stopword removal, lemmatization  
- TF-IDF vectorization for feature extraction  

---

## Features:  
- Cleaned and preprocessed reviews  
- Sentiment label assignment  
- Review statistics and visualizations:  
  - Word count distribution  
  - Sentiment distribution (histogram & pie chart)  
  - Monthly sentiment trends  
  - Word clouds and top negative words  

---

## Model:  
- **Random Forest Classifier** for sentiment prediction  
- Trained on TF-IDF features of review text  
- Separate models for Goboult and Flipflop datasets  
- Predictions: Positive / Negative  

---

## Streamlit Web App:  
- User selects dataset (Goboult / Flipflop)  
- User enters review text  
- Model predicts sentiment  
- Displays prediction in UI  
- Ngrok tunnel can provide a public URL  

---

## Installation:  
Required Python Libraries:  
- pandas  
- numpy  
- scikit-learn  
- nltk  
- matplotlib  
- seaborn  
- wordcloud  
- streamlit  
- pyngrok  

Install via pip:  
```bash
pip install -r requirements.txt
