import os
import certifi
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import ssl


ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# --- beállítások ---
DATA_ROOT = "/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Dataset_cats/Cat_vs_Dog"  # állítsd át
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "validation")
TEST_DIR = os.path.join(DATA_ROOT, "test")

IMG_SIZE = (300, 300)
BATCH_SIZE = 8
SEED = 123
EPOCHS = 15

# --- adatok betöltése ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels="inferred", label_mode="binary",
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, labels="inferred", label_mode="binary",
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, labels="inferred", label_mode="binary",
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED, shuffle=False)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# --- augmentáció ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),
    layers.RandomZoom(0.06),
])

# --- modell: MobileNetV2 ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

# --- callbacks ---
cbs = [
    callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
    callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
]

# --- edzés ---
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs)

# --- opcionális finomhangolás ---
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS//2, callbacks=cbs)

# --- értékelés a teszten ---
test_loss, test_acc, test_auc = model.evaluate(test_ds)
print("Test:", test_loss, test_acc, test_auc)

# --- minta predikció (képek sorrendje megegyezik a test_ds fájlrendelésével) ---
import numpy as np
for images, labels in test_ds.take(1):
    preds = model.predict(images)
    print("preds (0..1):", np.round(preds.flatten()[:10], 3))