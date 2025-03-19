import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

model_path = "models/text_trend_model.h5"  
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0 
    return img_array

def predict_fashion_trend(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)[0][0] 
    predicted_value = prediction[0][0]

    if predicted_value > 0.43:
        print(f"🚀 Trending Fashion Item ({predicted_value:.2f})")
    else:
        print(f"❌ Not a Trending Fashion Item ({predicted_value:.2f})")


def predict_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            print(f"\n📸 Predicting for {filename}...")
            predict_fashion_trend(img_path)

test_folder = "data/test_images/" 
predict_folder(test_folder)
