import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Flatten, MaxPool2D, Dense
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt

train_path = Path("/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Object_Detection_TF/train")
test_path = Path("/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Object_Detection_TF/test")
valid_path = Path("/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Object_Detection_TF/valid")

train=pd.read_csv("/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Object_Detection_TF/train/_annotations.csv")

train[['xmin', 'ymin', 'xmax', 'ymax']] = train[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
train.drop_duplicates(subset='filename', inplace=True, ignore_index=True)

def display_image(img, bbox_coorsd=[], pred_coords=[], norm=False):
    if norm:
        img *= 255.
        img = img.astype(np.uint8)

    img = img.copy() # OpenCV módosítás miatt kell

    if len(bbox_coorsd) == 4:
        xmin, ymin, xmax, ymax = bbox_coorsd
        img = cv.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)

    if len(pred_coords) == 4:
        xmin, ymin, xmax, ymax = pred_coords
        ing = cv.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 3)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # BGR to RGB átalakítás a helyes megjelenítéshez
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

def display_image_from_file(name, bbox_coorsd=[], path=train_path):
    img = cv.imread(str(path / name))
    display_image(img, bbox_coorsd=bbox_coorsd)

def display_from_dataframe(row, path=train_path):
    display_image_from_file(row['filename'], bbox_coorsd=(row.xmin, row.ymin, row.xmax, row.ymax), path=path)

def display_grid(df=train, n_item=3):
    plt.figure(figsize=(20, 10))

    rand_incidents = [np.random.randint(0, df.shape[0]) for _ in range(n_item)]

    for pos, index in enumerate(rand_incidents):
        plt.subplot(1, n_item, pos + 1)
        display_from_dataframe(df.loc[index, :])


""" # csak példa
display_image_from_file("beagle_25_jpg.rf.010a42a628cb4fb17c23f1129fffd2e3.jpg")

display_grid()
plt.show()

note: Helyesek az adatok és képekhez tartozó cimkék!
"""

def data_generator (df=train, batch_size=16, path=train_path):
    while True:
        images = np.zeros((batch_size, 640, 640, 3))
        bounding_box_coords = np.zeros((batch_size, 4))

        for i in range(batch_size):
            rand_index = np.random.randint(0, train.shape[0])
            row = df.loc[rand_index, :]
            images[i] = cv.imread(str(train_path/row.filename)) / 255.
            bounding_box_coords[i] = np.array([row.xmin, row.ymin, row.xmax, row.ymax])

        yield images, bounding_box_coords

# Teszt - sikeres

"""
example, label = next(data_generator(batch_size=1))
img = example['images'][0]
bbox_coords = label['coords'][0]

display_image(img, bbox_coorsd=bbox_coords, norm=True)
plt.show()
"""

input_ = Input(shape=[640, 640, 3], name ='images')


x = input_

for i in range(10):
    n_filters = 2**(i+3)
    x = Conv2D(n_filters, 3, activation='relu', padding='same' ) (x)
    x = BatchNormalization() (x)
    x = MaxPool2D(2, padding='same') (x)

x = Flatten() (x)
x = Dense(256, activation='relu') (x)
x = Dense(32, activation='relu') (x)
output = Dense(4, name='coords') (x)

model = tf.keras.Model(inputs=input_, outputs=output)
model.summary()

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(1e-3),
    metrics=['accuracy']
)

def test_model(model, datagen):
    example, label = next(datagen)

    X = example
    y = label
    
    pred_bbox = model.predict(X)[0]
    
    img = X[0]
    gt_coords = y[0]
    
    display_image(img, pred_coords=pred_bbox, norm=True)

def test(model):
    datagen = data_generator(batch_size=1)
    
    plt.figure(figsize=(15,7))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        test_model(model, datagen)    
    plt.show()
    
class ShowTestImages(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        test(self.model)

test(model)

with tf.device('/CPU:0'):
    _ = model.fit(
        data_generator(),
        epochs=10,
        steps_per_epoch=500,
        callbacks=[
            ShowTestImages(),
        ]
    )

    model.save('dog_detection.h5')