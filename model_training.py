
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
import string

stop_words = {'some', 'which', 'the', 'shouldn', 'her', 'up', 'm', 'won', "won't", 'been', 'hadn', 'himself', 'has', "aren't", 'wasn', "shan't", 'ours', 'very', "didn't", 'until', 'it', "needn't", "you've", 'wouldn', 'myself', 'don', 'after', 'same', 'themselves', 'out', 'ourselves', 'any', 'will', 'if', "mightn't", "it's", 'further', 'them', 'ma', "mustn't", 'how', 'were', "don't", 'herself', "weren't", 'am', 'll', 'then', 'its', "isn't", 'yourself', 'once', 'no', "shouldn't", 'o', 'i', 'those', 'd', 'because', 'own', 'his', 'theirs', 'him', 'doing', 'aren', 'couldn', 'to', 'so', "hadn't", 'during', 'over', 'here', 'does', 'why', 'other', 'just', 'at', 'doesn', 'while', 'having', 'mustn', 'under', 'for', 'should', 'a', 'isn', 'we', 'of', 'shan', 'be', 'nor', 'hasn', 'hers', 'me', 'this', 't', 'haven', 'with', 'only', 'an', 'what', 'against', 'can', 'she', 'or', 'such', 'as', 'through', "should've", 'mightn', 'there', 'are', 'into', 'above', 'whom', 'you', "you're", "that'll", 'down', 'most', 'your', 'needn', 'on', 'ain', 'that', 'their', 'itself', "doesn't", 'have', 'yourselves', 'being', 'these', 'now', 'was', 'did', "she's", 'all', "hasn't", 'but', 'each', 'our', 'again', 'didn', "you'd", "wasn't", "couldn't", 'in', 'when', 'than', 'before', 's', 'from', 've', 'is', 'yours', 'had', 'and', 'my', 'too', 're', 'between', 'they', 'both', 'who', 'below', 'where', 'he', 'do', 'not', 'by', "you'll", "wouldn't", 'few', 'off', 'more', "haven't", 'y', 'weren', 'about'}
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

data = {
    "label": ["ham", "spam", "ham", "spam"],
    "text": [
        "Hi there, how are you?",
        "Congratulations! You've won a prize. Call now!",
        "Are we still meeting today?",
        "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 12345"
    ]
}
df = pd.DataFrame(data)
df['cleaned'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label'].map({'ham': 0, 'spam': 1})
model = MultinomialNB()
model.fit(X, y)

pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
