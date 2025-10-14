import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Adatok betöltése
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizálás (0-255 → 0-1)
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
model.fit(x_train, y_train, epochs=5)
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Teszt veszteség: {loss:.4f}, Pontosság: {accuracy:.4f}")

plt.imshow(x_test[0], cmap='gray')
plt.title("Tesztkép: x_test[0]")
plt.axis('off')
plt.show()

prediction = model.predict(x_test[0:1])
predicted_label = tf.argmax(prediction[0]).numpy()
print(f"Predikált számjegy: {predicted_label}")
