import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences 
import pickle

# Check for GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    print("GPU Available:", gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, running on CPU.")

# Load datasets
myntra_df = pd.read_csv("data/Myntra Fasion Clothing.csv")


# Select relevant descriptions
myntra_desc = myntra_df["Description"].dropna().astype(str)


# Combine descriptions
all_descriptions = pd.concat([myntra_desc], ignore_index=True)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_descriptions)
sequences = tokenizer.texts_to_sequences(all_descriptions)


# Save tokenizer properly
with open("models/tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Tokenizer re-saved successfully!")


# Handling empty sequences
if len(sequences) == 0:
    raise ValueError("No valid text sequences found. Ensure data contains descriptions.")

max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length)  # Updated line
y = np.random.randint(0, 2, size=(len(X),))  # Fake trend labels (0 or 1)

# Define model
with tf.device("/GPU:0"):  
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_length),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model on GPU
with tf.device("/GPU:0"):  
    model.fit(X, y, epochs=10, batch_size=16)  

# Save model
model.save("models/text_trend_model.h5")
print("Text Model Training Complete!")
