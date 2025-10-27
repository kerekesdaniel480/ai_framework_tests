import os
import certifi
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import ssl
import matplotlib.pyplot as plt
import numpy as np


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
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs)

# --- értékelés a teszten ---
test_loss, test_acc, test_auc = model.evaluate(test_ds)
print("Test:", test_loss, test_acc, test_auc)

# 1) Accuracy görbe (train vs val)
def plot_accuracy(history):
    acc = history.history.get("accuracy") or history.history.get("acc")
    val_acc = history.history.get("val_accuracy") or history.history.get("val_acc")
    if acc is None or val_acc is None:
        print("Nincs accuracy adat a history-ban.")
        return
    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(7,4))
    plt.plot(epochs, acc, "b-", label="train accuracy")
    plt.plot(epochs, val_acc, "r--", label="val accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_accuracy(history)

# 2) 10 minta kép predikcióval
def show_sample_predictions(model, test_ds, n=10, class_names=None):
    # class_names: ha nincs megadva, próbáljuk meg lekérni a dataset-ből
    if class_names is None:
        try:
            class_names = test_ds.class_names
        except Exception:
            class_names = ["cat", "dog"]  # alapértelmezett
    # gyűjtsünk össze képeket és file-okat
    imgs = []
    trues = []
    preds = []
    for images, labels in test_ds:    # shuffle=False biztosítja, hogy sorrend következetes
        # iterálunk batchenként, kigyűjtjük amennyi kell
        p = model.predict(images, verbose=0).flatten()
        for img, lab, prob in zip(images.numpy(), labels.numpy(), p):
            imgs.append(img.astype(np.uint8))
            trues.append(int(lab))
            preds.append(float(prob))
            if len(imgs) >= n:
                break
        if len(imgs) >= n:
            break

    # kirajzolás
    cols = 5
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols*3, rows*3))
    for i in range(len(imgs)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(imgs[i])
        plt.axis("off")
        prob = preds[i]
        pred_label = class_names[1] if prob > 0.5 else class_names[0]
        true_label = class_names[trues[i]]
        title = f"P: {pred_label} ({prob:.2f})\nT: {true_label}"
        color = "green" if pred_label == true_label else "red"
        plt.title(title, color=color, fontsize=9)
    plt.tight_layout()
    plt.show()

show_sample_predictions(model, test_ds, n=10)