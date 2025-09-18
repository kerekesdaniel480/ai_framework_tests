import tensorflow as tf
import sys

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    'Dataset_cats/Cat vs Dog/train',
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'
)

validation_generator = datagen.flow_from_directory(
    'Dataset_cats/Cat vs Dog/validation',
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'
)


#print("TensorFlow version: ", tf.__version__, "\n")
print("Python version: ", sys.version, "\n")