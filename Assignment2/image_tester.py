
import PIL
import numpy as np
import tensorflow as tf
import tensorflow.keras as kb
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
from tensorflow.keras import backend



def print_image(path):
    ## load an image in PIL format
    original = load_img(path, target_size = (224, 224))
    print('PIL image size',original.size)
    plt.imshow(original)
    plt.show()
    return original

        
def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def get_best_worst_images(model, images, masks, num_images=3):
    predictions = model.predict(images)
    
    iou_scores = [calculate_iou(masks[i], predictions[i] > 0.5) for i in range(len(images))]
    
    sorted_indices = np.argsort(iou_scores)
    
    best_indices = sorted_indices[-num_images:]
    worst_indices = sorted_indices[:num_images]
    
    return best_indices, worst_indices


def display_images(j, images, masks, indices, title):
    plt.figure(figsize=(15, 7))
    plt.suptitle(title)
    
    for i, index in enumerate(indices):
        index = int(index)  # Convert index to integer
        
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[index])
        plt.title(f'Image {index}')
        
        plt.subplot(2, 3, i + 4)
        mask_to_display = masks[index]
        if mask_to_display.ndim == 3:  # Check if mask has 3 dimensions (RGB image)
            mask_to_display = mask_to_display[:, :, 0]  # Take one channel for grayscale
        plt.imshow(mask_to_display, cmap='gray')
        plt.title('Ground Truth')
    
    if j == 1:
        plt.savefig("3_best")
    elif j == 2:
        plt.savefig("3_worst")




def get_new_segmentation_map(model, path, image_width, image_height):
    # Load the new image
    new_image_path = path
    new_image = image.load_img(new_image_path, target_size=(image_width, image_height))
    new_image_array = image.img_to_array(new_image)
    new_image_array = np.expand_dims(new_image_array, axis=0) / 255.0  # Normalize the image

    # Predict the segmentation map
    predicted_mask = model.predict(new_image_array)

    # Threshold the predicted mask
    threshold = 0.5
    binary_mask = (predicted_mask > threshold).astype(np.uint8)

    # Visualize the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(new_image)
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(predicted_mask), cmap='gray', vmin=0, vmax=1)
    plt.title('Predicted Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(binary_mask), cmap='gray', vmin=0, vmax=1)
    plt.title('Binary Mask')

    plt.savefig("new_segmentation_map")


