import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()


import streamlit as st
import pickle
import string
from nltk.stem import PorterStemmer

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
        background-color: #f7f7f9;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextArea textarea {
        background-color: #fff !important;
        font-size: 16px !important;
    }
    </style>
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
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>📩 Spam Message Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter your message below to check if it's spam or not.</p>", unsafe_allow_html=True)

input_text = st.text_area("✍️ Type your message here:", height=150)

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
