# found invalid files during training, so this removes them
import os
import tensorflow as tf

def is_valid_jpeg(file_path):
    try:
        tf.io.decode_image(tf.io.read_file(file_path))
        return True
    except tf.errors.InvalidArgumentError:
        return False

def remove_invalid_jpeg_files(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and not is_valid_jpeg(file_path):
            os.remove(file_path)
            print(f"Removed invalid JPEG file: {file_path}")
            

def jpeg_remover(directory_path, subdirectories):

    # Specify the directory path containing the JPEG files
    directory = directory_path

    # Call the function to remove invalid JPEG files
    for directory in subdirectories:
        remove_invalid_jpeg_files(directory)