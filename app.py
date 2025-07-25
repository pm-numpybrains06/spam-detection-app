import nltk
from nltk.stem import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()

import streamlit as st
import pickle
import string
from nltk.stem import PorterStemmer

# Hardcoded stopwords
stop_words = {'some', 'which', 'the', 'shouldn', 'her', 'up', 'm', 'won', "won't", 'been', 'hadn', 'himself', 'has', "aren't", 'wasn', "shan't", 'ours', 'very', "didn't", 'until', 'it', "needn't", "you've", 'wouldn', 'myself', 'don', 'after', 'same', 'themselves', 'out', 'ourselves', 'any', 'will', 'if', "mightn't", "it's", 'further', 'them', 'ma', "mustn't", 'how', 'were', "don't", 'herself', "weren't", 'am', 'll', 'then', 'its', "isn't", 'yourself', 'once', 'no', "shouldn't", 'o', 'i', 'those', 'd', 'because', 'own', 'his', 'theirs', 'him', 'doing', 'aren', 'couldn', 'to', 'so', "hadn't", 'during', 'over', 'here', 'does', 'why', 'other', 'just', 'at', 'doesn', 'while', 'having', 'mustn', 'under', 'for', 'should', 'a', 'isn', 'we', 'of', 'shan', 'be', 'nor', 'hasn', 'hers', 'me', 'this', 't', 'haven', 'with', 'only', 'an', 'what', 'against', 'can', 'she', 'or', 'such', 'as', 'through', "should've", 'mightn', 'there', 'are', 'into', 'above', 'whom', 'you', "you're", "that'll", 'down', 'most', 'your', 'needn', 'on', 'ain', 'that', 'their', 'itself', "doesn't", 'have', 'yourselves', 'being', 'these', 'now', 'was', 'did', "she's", 'all', "hasn't", 'but', 'each', 'our', 'again', 'didn', "you'd", "wasn't", "couldn't", 'in', 'when', 'than', 'before', 's', 'from', 've', 'is', 'yours', 'had', 'and', 'my', 'too', 're', 'between', 'they', 'both', 'who', 'below', 'where', 'he', 'do', 'not', 'by', "you'll", "wouldn't", 'few', 'off', 'more', "haven't", 'y', 'weren', 'about'}

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing
stemmer = PorterStemmer()
def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("📩 Spam Message Detector")
input_text = st.text_area("Enter your message")

if st.button("Check"):
    cleaned = clean_text(input_text)
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)[0]
    st.write("🟢 This is a **Ham** message." if prediction == 0 else "🔴 This is a **Spam** message.")
