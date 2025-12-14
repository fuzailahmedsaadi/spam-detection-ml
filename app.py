import streamlit as st
import joblib
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

MODEL_PATH = "model/spam_classifier.pkl"

@st.cache_resource
def train_and_load_model():
    # Load dataset
    try:
        df = pd.read_csv("data/spam.csv", encoding="utf-8")
    except:
        df = pd.read_csv("data/spam.csv", encoding="latin-1")

    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df[["label", "text"]]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, _, y_train, _ = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("nb", MultinomialNB())
    ])

    model.fit(X_train, y_train)
    return model

model = train_and_load_model()

st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Spam Message Detection")
st.write("Enter a message to check whether it is **Spam** or **Ham**.")

message = st.text_area("✉️ Enter your message")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]
        if prediction == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is HAM (Not Spam)")
