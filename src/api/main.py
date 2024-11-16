import os
import sys
from abc import ABC, abstractmethod
from fastapi import FastAPI, Request, Form, HTTPException,File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(MAIN_DIR)
sys.path.append(MAIN_DIR)
MAIN_DIR = MAIN_DIR.split('Eye_disease')[0]
print(MAIN_DIR)
# Custom functions and methods
from utils.logging_utils import app_logger
from model.load_model import LoadPreTrainedModel
from preprocessing.process_image import Process_image
from prediction.prediction import Prediction
from PIL import Image
import base64
from pydantic import BaseModel
import tensorflow as tf
# Load pre-trained model and initialize FastAPI and templates
MODEL = LoadPreTrainedModel().call(model_path=f"{MAIN_DIR}/Eye_disease/models/best_model.keras")
APP = FastAPI()
TEMPLATES = Jinja2Templates(directory=f"{MAIN_DIR}/Eye_disease/src/api/templates")

@APP.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    """
    GET method to render the HTML form.
    """
    return TEMPLATES.TemplateResponse("index.html", {"request": request})

class ImageRequest(BaseModel):
    image: str  # Base64-encoded string


class PipelinePredictionAPI():
    def __init__(self):
        self.model = MODEL

    async def pipeline(self, image:tf.Tensor) -> str:
        """
        Predict status for the provided image.

        Args:
            image: the tensor that represents the image.

        Returns:
            str: The prediction result or an error message.
        """
        try:
            
           
            # Ensure the image input is valid
            
            if not isinstance(image, tf.Tensor):
                raise ValueError("Invalid input: text must be a non-empty string.")

            # process the image
            processed_image = Process_image.preprocess__image(image)
            app_logger.info(f"successfully processed image")

            # Make prediction
            prediction_result = Prediction().predict(model=self.model,image=processed_image)
            app_logger.info(f"Prediction result: {prediction_result}")
            return prediction_result

        except Exception as e:
            app_logger.error(f"An error occurred in the prediction pipeline: {e}")
            return "An error occurred while processing your request. Please try again."

# Instantiate the PipelinePredictionAPI
pipeline_api = PipelinePredictionAPI()

@APP.post("/predict", response_class=HTMLResponse)
async def predict(request:Request,Image_request:ImageRequest):
    """
    POST method to handle sentiment prediction.

    Args:
        request (Request): The FastAPI request object.
        Image_request: The input Object that contains the image to analyze.

    Returns:
        HTMLResponse: The rendered HTML response with prediction results or error messages.
    """
    # print(request)
    base64_str = Image_request.image
    
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    image_data = base64.b64decode(base64_str)
    print('decode image correctly')
    tensor = tf.io.decode_image(image_data, channels=3)
    print('converted to tensor ')
    # Run the prediction pipeline
    prediction_result =await pipeline_api.pipeline(tensor)
    print(prediction_result)
    return TEMPLATES.TemplateResponse("index.html",{"request":request,"result": prediction_result,})