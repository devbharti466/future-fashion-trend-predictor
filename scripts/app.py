import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import streamlit as st

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f" Error loading model: {e}")
        return None

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        img_array /= 255.0  
        return img_array
    except Exception as e:
        st.error(f" Error processing image: {e}")
        return None

def predict_fashion_trend(model, img_array):
    if img_array is None:
        return " Error in image processing"
    
    prediction = model.predict(img_array)
    
    st.write(f"📊 Prediction Raw Output: {prediction}")
    st.write(f"📏 Prediction Shape: {prediction.shape}")  

    if prediction.ndim > 1: 
        predicted_value = float(prediction.flatten()[0]) 
    else:
        predicted_value = float(prediction[0]) 

    if predicted_value > 0.43:
        return f"🚀 Trending Fashion Item ({predicted_value:.2f})"
    else:
        return f" Not a Trending Fashion Item ({predicted_value:.2f})"


def main():
    st.title("Fashion Trend Predictor")
    st.write("Upload an image to predict whether it's a trending fashion item!")
    
    model_path = "models/Trend-Model.h5" 
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
