import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Load the trained model
model_path = "models/text_trend_model.h5"
model = tf.keras.models.load_model(model_path)

# Load the tokenizer
tokenizer_path = "models/tokenizer.pkl"

try:
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)
    st.success("✅ Tokenizer loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading tokenizer: {e}")
    st.stop()

# Ensure correct max length (same as training)
MAX_LENGTH = 23  

# Function to predict trend from text description
def predict_fashion_trend(description):
    seq = tokenizer.texts_to_sequences([description])
    padded_seq = pad_sequences(seq, maxlen=MAX_LENGTH)
    prediction = model.predict(padded_seq)[0][0]

    if prediction > 0.5:
        return f"🚀 Trending Fashion Item ({prediction:.2f})"
    else:
        return f"❌ Not a Trending Fashion Item ({prediction:.2f})"

# Streamlit UI
st.title("👗 Fashion Trend Prediction")
st.write("Enter a fashion item description to predict if it's trending.")

# User Input
user_input = st.text_area("Enter Fashion Description:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a fashion description!")
    else:
        result = predict_fashion_trend(user_input)
        st.subheader("Prediction Result:")
        st.success(result)
