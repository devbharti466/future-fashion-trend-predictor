import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os

# Image Data Generator
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_data = datagen.flow_from_directory("data/images", target_size=(128, 128), batch_size=8, subset="training")
val_data = datagen.flow_from_directory("data/images", target_size=(128, 128), batch_size=8, subset="validation")

# CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
model.save("models/image_trend_model.h5")
print("Image Model Training Complete!")