import streamlit as st
import pickle

# Load the model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("Sentiment Analysis Web App")
st.write("Enter a review below to analyze its sentiment.")

# User input for review
review_text = st.text_area("Enter your review:", "")

# Predict sentiment when button is clicked
if st.button("Analyze Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter a review before submitting.")
    else:
        # Preprocess the input
        input_vector = vectorizer.transform([review_text])
        
        # Predict sentiment
        prediction = model.predict(input_vector)[0]
        
        # Map prediction to sentiment
        sentiment_map = {1: '😊 Positive', 0: '😐 Neutral', -1: '😠 Negative'}
        sentiment = sentiment_map[prediction]
        
        # Display result
        st.subheader(f"Sentiment: {sentiment}")
