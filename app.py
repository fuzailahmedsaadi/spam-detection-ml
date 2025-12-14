import streamlit as st
import joblib

# Load trained model
model = joblib.load("model/spam_classifier.pkl")

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
