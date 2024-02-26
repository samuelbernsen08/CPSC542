import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import tensorflow.keras as kb
from tensorflow.keras import backend
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental import preprocessing


def model_creator(batch_size = 32, image_width = 224, image_height = 224, epochs = 10, directory = "./CarBodyStyles"):

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels = "inferred",
        validation_split=0.3,
        image_size = (image_height, image_width),
        subset="both",
        seed=542,
        batch_size=batch_size,
    )
    class_names = train_ds.class_names

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)# early stopping

    model2 = tf.keras.applications.vgg16.VGG16(
        weights = "imagenet",
        include_top = False, # only including the convolution and pooling layers, not the dense FF layers
        input_shape = (image_width, image_height,3)
    )

    model2.trainable = False
    #model2.summary()

    inputs = kb.Input(shape=(image_width, image_height, 3)) # 3 channels = RGB image
    x = kb.applications.vgg16.preprocess_input(inputs)
    # Data augmentation layer
    data_augmentation = tf.keras.Sequential([
        preprocessing.RandomRotation(0.06),  # Randomly rotate the image by up to 6 degrees
        preprocessing.RandomZoom(0.05),  # Randomly zoom into the image by up to 5%
    ])

    x = data_augmentation(inputs)
    x = model2(x) # sending images through the pre-trained network
    x = kb.layers.Flatten()(x)
    x = kb.layers.Dense(135, kernel_regularizer = "l1")(x)
    x = kb.layers.Dropout(0.4)(x)
    x = kb.layers.Dense(111, kernel_regularizer = "l1")(x)
    x = kb.layers.Dropout(0.5)(x)
    x = kb.layers.Dense(30, kernel_regularizer = "l1")(x)
    outputs = kb.layers.Dense(7, activation="softmax")(x)

    custom_ff = kb.Model(inputs, outputs)
    custom_ff.compile(loss="sparse_categorical_crossentropy",
                optimizer="Adam",
                metrics=["accuracy"])

    history = custom_ff.fit(
    train_ds,
    validation_data= val_ds,
    epochs=epochs,
    callbacks = [callback], # for regularization
    initial_epoch = 0
    )

    #custom_ff.summary()
    custom_ff.save("./model_directory")
    return history


def get_class_names(directory, batch_size = 32, image_width = 224, image_height = 224):
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    directory = directory,
    labels = "inferred",
    validation_split=0.3,
    image_size = (image_height, image_width),
    subset="both",
    seed=542,
    batch_size=batch_size,
    )
    return train_ds.class_names