import os
import sys
# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)
from abc import ABC, abstractmethod
from typing import Any, Union
from utils.logging_utils import app_logger
import tensorflow as tf


class Process_image():

    def preprocess__image(path, data_augmentation=False):
        """
        Preprocess a  image for prediction.

        Args:
            path (str): Path to the image.
            data_augmentation (bool): Whether to apply data augmentation.

        Returns:
            tf.Tensor: Preprocessed image tensor.
        """
        # Read the image file
        image = tf.io.read_file(path)

        # Extract file extension
        file_extension = tf.strings.split(path, '.')[-1]

        # Decode based on file extension using tf.cond
        def decode_jpeg():
            return tf.image.decode_jpeg(image, channels=3)

        def decode_png():
            return tf.image.decode_png(image, channels=3)

        def decode_bmp():
            return tf.image.decode_bmp(image, channels=3)

        def decode_gif():
            # Decode GIF and take the first frame
            return tf.squeeze(tf.image.decode_gif(image), axis=0)

        # Handle each format
        image = tf.cond(tf.math.equal(file_extension, 'jpg'), decode_jpeg,
                lambda: tf.cond(tf.math.equal(file_extension, 'jpeg'), decode_jpeg,
                lambda: tf.cond(tf.math.equal(file_extension, 'png'), decode_png,
                lambda: tf.cond(tf.math.equal(file_extension, 'bmp'), decode_bmp,
                lambda: tf.cond(tf.math.equal(file_extension, 'gif'), decode_gif,
                decode_jpeg)))))


        # Resize and normalize
        image = tf.image.resize(image, [256, 256])
        image = image / 255.0  # Normalize to [0, 1] range

        # Apply data augmentation if requested
        if data_augmentation:
            # Randomly flip the image horizontally
            image = tf.image.random_flip_left_right(image)

            # Randomly flip the image vertically
            image = tf.image.random_flip_up_down(image)

            # Randomly rotate the image
            image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

            # Randomly adjust brightness
            image = tf.image.random_brightness(image, max_delta=0.1)

            # Randomly zoom in
            image = tf.image.resize_with_crop_or_pad(image, 266, 266)  # Zoom in slightly
            image = tf.image.random_crop(image, size=[256, 256, 3])

            # Randomly adjust contrast
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        # Add batch dimension
        image = tf.expand_dims(image, axis=0)
        
        return image
        

if __name__ == "__main__":
    iProcessImage = Process_image()
    clearn_text = iProcessImage.preprocess__image('')