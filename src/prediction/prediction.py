from abc import ABC, abstractmethod
from typing import Optional, Any
from keras._tf_keras.keras.models import load_model  # type: ignore
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences # type: ignore
from keras._tf_keras import keras
import os
import sys
from typing import Union
import numpy as np
from PIL import Image


# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.logging_utils import app_logger



class Prediction():
    def predict(self, model: keras.Model,image: tf.Tensor) -> str:
        """
        Predicts Eye status if it's normal or has a disease from the input image using a pre-trained model.

        Args:
            model (keras.Model): The pre-trained model used for prediction.
            tokenizer (tf.keras.preprocessing.text.Tokenizer): The tokenizer used to preprocess the text.
            text (str): The cleaned input text to predict sentiment on.

        Returns:
            str: The predicted status and confidence score.

        Raises:
            TypeError: If inputs are not of the expected types.
            Exception: For any other errors encountered during prediction.
        """
        try:
            # Check model
            if not isinstance(model, keras.Model):
                raise TypeError("Provided model is not a valid Keras model.")

            # Check text input
            if not isinstance(image, tf.Tensor):
                raise TypeError("Provided image input is not a tf.Tensor.")

            # Predict sentiment
            prediction = model.predict(image)

            class_labels = {
                0: 'cataract',
                1: 'diabetic_retinopathy',
                2: 'glaucoma',
                3: 'normal'
            }
            # Get the predicted class index
            predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
            

            # Get the string label for the predicted class and confidence
            predicted_label = class_labels[predicted_class_index]
            confidence = prediction[0][predicted_class_index]

            app_logger.info(f"Prediction made successfully for input text.")

            return f"Statues of Eye Image is: {predicted_label}, Confidence: {confidence:.2f}"

        except TypeError as type_err:
            app_logger.error(f"Type error during prediction: {type_err}")
            return "Error: Invalid input type for model, tokenizer, or text."

        except Exception as error:
            app_logger.error(f"An unexpected error occurred during prediction: {error}")
            return "Error: Prediction failed due to an unexpected issue."

from model.load_model import LoadPreTrainedModel

if __name__ == "__main__":
    model = LoadPreTrainedModel().call(model_path="/workspaces/Sentiment-Analysis/models/final_sentiment_model.h5")
    predict = Prediction().predict(model=model,image=tf.Tensor())

    print(predict)