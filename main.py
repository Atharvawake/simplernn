import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load IMDB word index and model
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
model = load_model('simple_rnn_imdb.h5')

# Function to decode reviews from encoded format
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Function to predict sentiment
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify whether it is positive or negative.')

# Fix typo in text input function
user_input = st.text_area('Movie Review')

# Button to classify sentiment
if st.button('Classify'):
    if user_input.strip():  # Check if input is not empty
        sentiment, score = predict_sentiment(user_input)
        st.write(f'**Sentiment:** {sentiment}')
        st.write(f'**Prediction Score:** {score:.4f}')
    else:
        st.warning('Please enter a movie review before classification.')
