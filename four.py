import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

x_train = pad_sequences(x_train, maxlen = 200)
x_test = pad_sequences(x_test, maxlen = 200)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim = 10000, output_dim = 128, input_length = 200),
    tf.keras.layers.LSTM(128, return_sequences=False),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score
y_pred = (model.predict(x_test)>0.5).astype("int32")
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

from tensorflow.keras.preprocessing.text import Tokenizer
def preprocess_input(text):
  tokenizer = Tokenizer(num_words = 10000)
  tokenizer.fit_on_texts([text])
  sequence = tokenizer.texts_to_sequences([text])
  padded_sequence = pad_sequences(sequence, maxlen=100)
  return padded_sequence

user_input = input("Enter a movie review: ")

processed_input = preprocess_input(user_input)

prediction = model.predict(processed_input)
sentiment = "positive" if prediction[0][0] > 0.5 else "Negative"
print(f"Predicted Sentiment:{sentiment} (Probability : {prediction[0][0]:.2f})")
