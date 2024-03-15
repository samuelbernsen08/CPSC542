import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import tensorflow.keras as kb
from tensorflow.keras import backend
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental import preprocessing
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#import cv2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage import io, color
import skimage.filters as filters

import image_tester
import glob

def model_creator(batch_size = 128, image_width = 224, image_height = 224, epochs = 10):

    # Set the paths to your images and masks
    images_path =  "./WaterBodiesDataset/Images/*.jpg"
    masks_path =  "./WaterBodiesDataset/Masks/*.jpg"

    # Get the list of image and mask file paths
    image_files = glob.glob(images_path)
    mask_files = glob.glob(masks_path)

    # Load images and masks as NumPy arrays
    images = [tf.keras.preprocessing.image.load_img(image, target_size=(image_width, image_height)) for image in image_files]
    masks = [tf.keras.preprocessing.image.load_img(mask, target_size=(image_width, image_height), color_mode='grayscale') for mask in mask_files]

    images = np.array([tf.keras.preprocessing.image.img_to_array(image) for image in images])
    masks = np.array([tf.keras.preprocessing.image.img_to_array(mask) for mask in masks])

    # Normalize masks and images
    masks = masks / 255.0
    images = images / 255.0

    # Used for random forest:

    # threshold = 0.5  # Adjust the threshold value as needed
    # masks = binarize_masks(masks, threshold)

    # Split data into training and validation sets
    train_images, val_images, train_masks, val_masks = train_test_split(images, masks, train_size=0.7, random_state=542)

    # print(masks[1:5])
    # early stopping for regularization
    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)# early stopping

    # using DataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.1,
        shear_range=0.25,
        zoom_range=0.21,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Random forest classifier:
    # RF = RF_architecture(images[1:150], masks[1:150])

    # display_segmented_masks(RF)


    train_generator = datagen.flow(train_images, train_masks, batch_size=batch_size)

    model = model_architecture(image_height=image_height, image_width=image_width)

    optimizer = kb.optimizers.Adam(learning_rate=0.005)  # Adjust the learning rate
    model.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])

    history = model.fit(
    train_generator,
    validation_data = (val_images, val_masks),
    batch_size = batch_size,
    epochs=epochs,
    initial_epoch = 0
    )

    model.save("./model_directory")

    # Get the best and worst performing indices
    best_indices, worst_indices = image_tester.get_best_worst_images(model, val_images, val_masks, num_images=3)

    image_tester.display_images(1, val_images, val_masks, best_indices, "Best Images")
    image_tester.display_images(2, val_images, val_masks, worst_indices, "Worst Images")


    return history

    

def model_architecture(image_width, image_height):
    inputs = kb.Input((image_width, image_height, 3))
    # U Net

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.25)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.3)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.25)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9) # 1x1 convolution to combine all channels for each pixel
    model = Model(inputs=[inputs], outputs=[outputs])


    return model


def RF_architecture(images, masks):
    # Reshape images and masks
    X = np.array(images).reshape(len(images), -1)
    y = np.array(masks).reshape(len(masks), -1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=2, random_state=42)
    # Train the model
    rf_model.fit(X_train, y_train)
    # Predict masks for test data
    y_pred = rf_model.predict(X_test)

    # Reshape predicted masks
    segmented_masks = []
    for pred in y_pred:
        try:
            # Reshape predicted mask to match the shape of the original image
            segmented_mask = pred.reshape((224, 224))
            segmented_masks.append(segmented_mask)
        except ValueError:
            # If the predicted mask cannot be reshaped, handle the error gracefully
            print("Error: Cannot reshape predicted mask. Skipping...")
            continue

    return segmented_masks

def display_segmented_masks(segmented_masks):
    plt.figure(figsize=(10, 5))
    num_masks = len(segmented_masks)
    for i in range(num_masks):
        plt.subplot(1, num_masks, i + 1)
        plt.imshow(segmented_masks[i], cmap='gray')
        plt.title(f'Segmented Mask {i}')
        plt.axis('off')
    plt.show()


def binarize_masks(masks, threshold):
    binary_masks = []
    for mask in masks:
        # Apply thresholding to binarize the mask
        binary_mask = mask > threshold
        binary_masks.append(binary_mask)
    return binary_masks
