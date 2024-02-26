
import PIL
import numpy as np
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import tensorflow.keras as kb
from tensorflow.keras import backend
import tensorflow as tf

def print_image(path):
    ## load an image in PIL format
    original = load_img(path, target_size = (224, 224))
    print('PIL image size',original.size)
    plt.imshow(original)
    plt.show()
    return original

def get_prediction(path, model, class_names):
    #image = print_image(path) # uncomment if the output actually shows
    image = load_img(path, target_size = (224, 224))
    numpy_image = img_to_array(image)
    image_batch = np.expand_dims(numpy_image, axis = 0)
    processed_image = kb.applications.vgg16.preprocess_input(image_batch.copy())
    prediction = model.predict(processed_image)
    # prediction
    score = tf.nn.softmax(prediction[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
    
'''
path = './TestingImages/2017-volkswagen-golf-gti-sport-review.jpg'
loaded_model = tf.keras.models.load_model("./model_directory")
classes = model_creator.get_class_names(directory = "./CarBodyStyles")
print("This should be a hatchback.")
image_tester.get_prediction(path = path, model = loaded_model, class_names = classes)
'''

def test_image(path, model, classes, correct_class):
    print("This should be a", correct_class, ".")
    get_prediction(path, model, classes)
    print() #newline
    

def test_multiple_images(image_dict, model, classes):
    keys = list(image_dict.keys())
    for i in range(len(image_dict)):
        path = keys[i]
        test_image(path, model, classes, image_dict[path])