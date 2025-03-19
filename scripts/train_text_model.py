import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences 
import pickle

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

myntra_df = pd.read_csv("data/Myntra Fasion Clothing.csv")

myntra_desc = myntra_df["Description"].dropna().astype(str)

all_descriptions = pd.concat([myntra_desc], ignore_index=True)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_descriptions)
sequences = tokenizer.texts_to_sequences(all_descriptions)


with open("models/tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Tokenizer re-saved successfully!")

if len(sequences) == 0:
    raise ValueError("No valid text sequences found. Ensure data contains descriptions.")

max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length) 
y = np.random.randint(0, 2, size=(len(X),)) 

with tf.device("/GPU:0"):  
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_length),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

with tf.device("/GPU:0"):  
    model.fit(X, y, epochs=10, batch_size=16)  

model.save("models/text_trend_model.h5")
print("Text Model Training Complete!")
