import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import streamlit as st

#Load the trained model safely
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

#Function to preprocess image before prediction
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize
        return img_array
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None

#Function to predict if fashion item is trending
def predict_fashion_trend(model, img_array):
    if img_array is None:
        return "❌ Error in image processing"
    
    prediction = model.predict(img_array)
    
    st.write(f"📊 Prediction Raw Output: {prediction}")  # Debugging Output
    st.write(f"📏 Prediction Shape: {prediction.shape}")  # Debugging Output

    if prediction.ndim > 1:  # If prediction is multi-dimensional
        predicted_value = float(prediction.flatten()[0])  # Flatten to 1D and get first value
    else:
        predicted_value = float(prediction[0])  # Directly get the value

    if predicted_value > 0.43:
        return f"🚀 Trending Fashion Item ({predicted_value:.2f})"
    else:
        return f"❌ Not a Trending Fashion Item ({predicted_value:.2f})"


# ✅ Streamlit UI
def main():
    st.title("👕 Fashion Trend Predictor")
    st.write("Upload an image to predict whether it's a trending fashion item!")
    
    model_path = "models/Trend-Model.h5"  # Change this path if needed
    model = load_model(model_path)
    if model is None:
        return
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("Analyzing image...")
        
        img_array = preprocess_image(uploaded_file)
        result = predict_fashion_trend(model, img_array)
        st.success(result)

if __name__ == "__main__":
    main()
