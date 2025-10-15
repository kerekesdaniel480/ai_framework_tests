import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


train_datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

train_generator = train_datagen.flow_from_directory(
    '/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Dataset_cats/Cat_vs_Dog/train',
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    '/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Dataset_cats/Cat_vs_Dog/validation',
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical'
)

model = Sequential([
    Input(shape=(300, 300, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

img_path = '/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Dataset_cats/Cat_vs_Dog/train/dog/br-conf-0.15398428-t-1692019699061.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
predicted_class = tf.argmax(prediction, axis=1).numpy()

class_names = train_generator.class_indices
reverse_class_names = {v: k for k, v in class_names.items()}
print("Predikció:", reverse_class_names[predicted_class[0]])

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stop, lr_scheduler]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Pontosság (accuracy) ábrázolása
plt.figure(figsize=(10, 5))
plt.plot(epochs, acc, 'bo-', label='Tanító pontosság')
plt.plot(epochs, val_acc, 'ro-', label='Validációs pontosság')
plt.title('Modell pontosság')
plt.xlabel('Epoch')
plt.ylabel('Pontosság')
plt.show()

max_train_acc = max(history.history['accuracy'])
max_val_acc = max(history.history['val_accuracy'])

print(f"Maximális tanító pontosság: {max_train_acc * 100:.2f}%")
print(f"Maximális validációs pontosság: {max_val_acc * 100:.2f}%")