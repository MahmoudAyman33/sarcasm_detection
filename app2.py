import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from keras import regularizers
nltk.download('stopwords')
vocab_size = 10000
embedding_dim = 16
lstm_units = 64 # number of units in the LSTM layer
model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
model_lstm.add(Bidirectional(LSTM(units=lstm_units, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=regularizers.l2(0.01))))
model_lstm.add(Dense(units=1, activation='sigmoid'))
# compile the model
model_lstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model_lstm.load_weights('sarcasm_detection_model.h5')

with open('tokenizer.json', 'r') as f:
    tokenizer_json = f.read()

# Load the tokenizer from the JSON content
tokenizer = tokenizer_from_json(tokenizer_json)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(headline):
    # Remove extra spaces
    headline = re.sub(r'\s+', ' ', headline)
    # Remove mentions (@username)
    headline = re.sub(r'@[\w-]+', '', headline)
    # Remove URLs
    headline = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', headline)
    # Remove punctuations, numbers, and convert to lowercase
    headline = re.sub(r'[^a-zA-Z]', ' ', headline).lower()
    # Tokenize
    words  = headline.split()
    # Remove stopwords and apply stemming
    words  = [stemmer.stem(word) for word in words if word not in stop_words]
    # Join the words back into a string
    processed_headline = ' '.join(words)
    return processed_headline

# Streamlit application
st.title("Sarcasm Detection")
st.write('Enter Any Tweet')

# Text input from user
custom_text = st.text_area("Text to analyze")
if st.button("Enter"):
    if custom_text.strip():
        # Preprocess the input text
        preprocessed_input = preprocess(custom_text)
        # Predict sentiment
        tokenized = tokenizer.texts_to_sequences([preprocessed_input])
        prediction = model_lstm.predict(tokenized)
        # Get the class with the highest probability
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        st.write('Sarcastic Propability: ',confidence) 
