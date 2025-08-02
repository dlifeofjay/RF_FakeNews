import sys
import streamlit as st
import joblib
import pandas


model = joblib.load('RFFN_model.joblib')
vectorizer = joblib.load('RFFN_vectorizer.joblib')


def clean_text(text):
    text = text.lower().strip()
    text = text.replace('?', '')
    text = text.replace('.', '')
    text = text.replace('-', ' ')
    text = text.replace(',', '')
    text = text.replace("'", '')
    text = text.replace('"', '')
    text = text.replace('!', '')
    text = text.replace(';', '')
    text = text.replace(':', '')
    text = text.replace('&', '')
    text = text.replace('_', ' ')
    return text


st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector Using Random Forest Classifier")
st.markdown("Enter a news article or headline to check if it's **Real** or **Fake**.")

user_input = st.text_area("📝 News Text", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        prediction_proba = model.predict_proba(vectorized_text)[0]

        confidence = prediction_proba[prediction] * 100
        label = "🟢 Real News ✅" if prediction == 1 else "🔴 Fake News ❌"
        st.subheader("Prediction Result:")
        st.success(f"{label}\n\n🧠 Confidence Score: {confidence:.2f}%")
        
