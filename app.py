import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()


import streamlit as st
import pickle
import string
import json

from nltk.stem import PorterStemmer
from streamlit_lottie import st_lottie


def load_lottie():
    with open("spam_animation1.json", "r") as f:
        return json.load(f)

st_lottie(load_lottie(), height=250)


# 🌐 Hardcoded stopwords (safe for deployment)
stop_words = {'some', 'which', 'the', 'shouldn', 'her', 'up', 'm', 'won', "won't", 'been', 'hadn', 'himself', 'has', "aren't",
              'wasn', "shan't", 'ours', 'very', "didn't", 'until', 'it', "needn't", "you've", 'wouldn', 'myself', 'don', 'after',
              'same', 'themselves', 'out', 'ourselves', 'any', 'will', 'if', "mightn't", "it's", 'further', 'them', 'ma',
              "mustn't", 'how', 'were', "don't", 'herself', "weren't", 'am', 'll', 'then', 'its', "isn't", 'yourself', 'once',
              'no', "shouldn't", 'o', 'i', 'those', 'd', 'because', 'own', 'his', 'theirs', 'him', 'doing', 'aren', 'couldn',
              'to', 'so', "hadn't", 'during', 'over', 'here', 'does', 'why', 'other', 'just', 'at', 'doesn', 'while', 'having',
              'mustn', 'under', 'for', 'should', 'a', 'isn', 'we', 'of', 'shan', 'be', 'nor', 'hasn', 'hers', 'me', 'this',
              't', 'haven', 'with', 'only', 'an', 'what', 'against', 'can', 'she', 'or', 'such', 'as', 'through', "should've",
              'mightn', 'there', 'are', 'into', 'above', 'whom', 'you', "you're", "that'll", 'down', 'most', 'your', 'needn',
              'on', 'ain', 'that', 'their', 'itself', "doesn't", 'have', 'yourselves', 'being', 'these', 'now', 'was', 'did',
              "she's", 'all', "hasn't", 'but', 'each', 'our', 'again', 'didn', "you'd", "wasn't", "couldn't", 'in', 'when',
              'than', 'before', 's', 'from', 've', 'is', 'yours', 'had', 'and', 'my', 'too', 're', 'between', 'they', 'both',
              'who', 'below', 'where', 'he', 'do', 'not', 'by', "you'll", "wouldn't", 'few', 'off', 'more', "haven't", 'y',
              'weren', 'about'}

# 🎨 Page setup and styling
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #eb86c7;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextArea textarea {
        background-color: #eb86c7 !important;
        font-size: 16px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: 'Bebas Neue', sans-serif;
        }
    </style>
    <h1 style='color:#3366cc;'>SpamShield - Smart Spam Detector</h1>
""", unsafe_allow_html=True)



# 🧠 Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🧽 Preprocessing function
stemmer = PorterStemmer()
def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# 🖼️ Main UI
st.markdown("<p style='text-align: center;'>Enter your message below to check if it's spam or not.</p>", unsafe_allow_html=True)
st.markdown("📌 Try example messages:")

examples = {
    "Spam": "Congratulations! You’ve won a $1000 gift card. Click here to claim.",
    "Ham": "Hey, are we still on for dinner tonight?"
}

# Use a session state variable to persist button choice
if "example_text" not in st.session_state:
    st.session_state.example_text = ""

col1, col2 = st.columns(2)
with col1:
    if st.button("🎁 Spam Example"):
        st.session_state.example_text = examples["Spam"]
with col2:
    if st.button("💬 Ham Example"):
        st.session_state.example_text = examples["Ham"]

# Show the selected example or allow user input
input_text = st.text_area("✍️ Type your message here:", value=st.session_state.example_text, height=150)


if st.button("🔍 Analyze"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        cleaned = clean_text(input_text)
        vect_text = vectorizer.transform([cleaned])
        prediction = model.predict(vect_text)[0]

        if prediction == 0:
            st.success("✅ This is a **HAM** message. Safe and normal.")
        else:
            st.error("🚫 This is a **SPAM** message. Be cautious!")

# 📌 Sidebar
st.sidebar.title("📘 About")
st.sidebar.info("""
This is a simple Spam Detection app built using:
- Python 🐍
- Scikit-learn 🤖
- Streamlit 🌐
- NLTK ✂️

Developed by **Prajukta Mandal** 💡
""")

# 📎 Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
