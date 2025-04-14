import streamlit as st
import pickle
import re
from bs4 import BeautifulSoup

# Preprocess text
def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().strip()
    return text

# Load the model and vectorizer
def load_model():
    with open('news_classifier_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Predict category
def predict_news_category(text, model, vectorizer):
    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)
    return prediction[0]

# Streamlit UI
st.set_page_config(page_title="News Article Classification", layout="wide")

# Title and links
st.title("📰 News Article Classification")

col1, col2 = st.columns([8, 2])
# with col2:
#     st.markdown('[📁 Dataset Link]()')
#     st.markdown('[💻 GitHub Link](https://github.com)')

# Problem Statement
st.write("""
## Problem Statement
The primary objective of this project is to build a classification model that can automatically categorize news articles into different predefined categories. 
The model will be trained using a labeled dataset of news articles and will output the most likely category (e.g., sports, politics, or technology) for any given article.

""")

# Example input
st.write("### Example Articles:")
st.write(""" *New research shows that regular exercise can improve mental health by reducing stress and anxiety levels.*""")
st.write(""" *The government announced new policies to address climate change and reduce carbon emissions.*""")

# User input
st.write("### Enter a News Article for Classification")
user_input = st.text_area("", "Type your article here...")

if st.button("Predict Category"):
    model, vectorizer = load_model()
    prediction = predict_news_category(user_input, model, vectorizer)
    st.write(f"### Predicted Category: 🏷️ {prediction}")

st.write("\nThank you for using our News Article Classification App! 😊")
