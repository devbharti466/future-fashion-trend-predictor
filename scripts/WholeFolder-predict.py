import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

#Load the trained model (update path if needed)
model_path = "models/text_trend_model.h5"  # Change if stored elsewhere
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

#Function to preprocess image before prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # Resize image
    img_array = image.img_to_array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize like training data
    return img_array

#Function to predict if the fashion item is trending
def predict_fashion_trend(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)[0][0]  # Extract single prediction value
    # Extract single value from array
    predicted_value = prediction[0][0]

    if predicted_value > 0.43:
        print(f"🚀 Trending Fashion Item ({predicted_value:.2f})")
    else:
        print(f"❌ Not a Trending Fashion Item ({predicted_value:.2f})")

#Predict for a single image
# image_path = r"C:\Users\ASUS\OneDrive\Desktop\css\fashion-trend-prediction\af4540c278dee86b2568581dd3a1aea6.jpg"  # Change this to your test image path
# predict_fashion_trend(image_path)

#Predict for multiple images in a folder
def predict_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            print(f"\n📸 Predicting for {filename}...")
            predict_fashion_trend(img_path)

# Example usage
test_folder = "data/test_images/"  # Change this to your folder path
predict_folder(test_folder)
