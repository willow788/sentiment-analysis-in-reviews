# app.py
import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load('nb_model.pkl')
vectorizer = joblib.load('tfidf.pkl')

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Streamlit UI
st.title("IMDb Sentiment Analysis")
user_input = st.text_area("Enter a movie review:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    st.write("Sentiment:", prediction)
