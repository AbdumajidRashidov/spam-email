import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error("Required model files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
    st.stop()

st.title("Email/SMS Spam Detector")
st.markdown("Enter your message below and click 'Predict' to check if it's spam or not.")
input_sms = st.text_area("Enter the message")

if st.button('Predict', key='email'):
    if input_sms == "":
        st.subheader("Please enter a message")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Spam")
            st.subheader(f"Confidence: {(model.predict_proba(vector_input)[0][1] * 100):.2f}%")
        else:
            st.header("Not Spam")
            st.subheader(f"Confidence: {(model.predict_proba(vector_input)[0][0]*100):.2f}%")
