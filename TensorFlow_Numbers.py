import random
import ssl
import certifi
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

tf.random.set_seed(42) 
np.random.seed(42)
random.seed(42)

# Adatok betöltése
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 28x28 képek → 784 vektor
    tf.keras.layers.Dense(128, activation='relu'),  # ReLU aktivációs függvény
    tf.keras.layers.Dropout(0.2),                   # Regularizáció
    tf.keras.layers.Dense(10, activation='softmax') # 10 osztály (0–9)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Teszt veszteség: {loss:.4f}, Pontosság: {accuracy:.4f}")

# Teszt képek
plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.axis('off')
    prediction = model.predict(x_test[i:i+1])
    predicted_label = tf.argmax(prediction[0]).numpy()
    plt.title(f"Pred: {predicted_label}")
plt.suptitle("Első 5 tesztkép és predikció")
plt.tight_layout()
plt.show()

# Tanulási görbék 
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
