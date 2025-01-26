X_OR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y_OR = np.array([[0], [1], [1], [1]], dtype=np.float32)
X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_XOR = np.array([[0], [1], [1], [0]], dtype=np.float32)

import tensorflow as tf
def train_perceptron(X, Y, epochs=100, learning_rate=0.1):
  model = tf.keras.Sequential([
  tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
  ])
  model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=
  learning_rate),loss='binary_crossentropy',
  metrics=['accuracy'])
  model.fit(X, Y, epochs=epochs, verbose=0)
  return model

model_OR = train_perceptron(X_OR, Y_OR, epochs=500, learning_rate=0.5)
loss_OR, accuracy_OR = model_OR.evaluate(X_OR, Y_OR)
print(f"OR Gate Accuracy: {accuracy_OR}")
model_XOR = train_perceptron(X_XOR, Y_XOR, epochs=1000, learning_rate=0.8)
loss_XOR, accuracy_XOR = model_XOR.evaluate(X_XOR, Y_XOR)
print(f"XOR Gate Accuracy: {accuracy_XOR}")

input1 = 0
input2 = 0
user_input = np.array([[input1, input2]])
prediction = model_OR.predict(user_input)
if prediction > 0.5:
  print("The model predicts 1 for your input.")
else:
  print("The model predicts 0 for your input.")
