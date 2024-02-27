Samuel Bernsen

ID: 2367195

In-class collaborators: Kelsey Hawkins, Joe Ellis, Dylan McIntosh

Resources: 
https://www.tutorialspoint.com/keras/keras_real_time_prediction_using_resnet_model.htm
https://stackoverflow.com/questions/72479044/cannot-import-name-load-img-from-keras-preprocessing-image
https://www.kaggle.com/code/alperkoc/data-augmentation-vgg16-cnn
https://www.tensorflow.org/tutorials/images/classification
https://www.tensorflow.org/tutorials/load_data/images
https://www.tensorflow.org/tutorials/images/data_augmentation
https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict

Github repo: https://github.com/samuelbernsen08/CPSC542

Data gathered from: https://www.kaggle.com/datasets/darshan1504/car-body-style-dataset

Notes:
During runtime, you will see "invalid sRGB profile" for many of the images. This warning did not cause problems during training.
Running this project:
To run this project, download all the .py files and run them in your server environment.
If you want to change the number of epochs used during training, change the "epochs" variable in program.py.
The first time you run the program, answer "Y" to the initial prompt for training the model.
Once the model has been run, that model can be loaded for future inferencing by entering "N."
If you want to test on different images than the ones I have provided, change the strings in the "image_paths" list in program.py and change the strings in the "test_image_paths_and_labels" dictionary to match your path and correct label.
