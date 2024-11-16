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


class PipelinePredictionAPI():
    def __init__(self):
        self.model = MODEL

    async def pipeline(self, file: str) -> str:
        """
        Predict status for the provided image.

        Args:
            text (str): the image path.

        Returns:
            str: The prediction result or an error message.
        """
        try:
            # Read the image file
            # contents = await file.read()
            # image = Image.open(io.BytesIO(contents)).convert("RGB")
            # Ensure the text input is valid
            print(file)
            if not isinstance(file, str):
                raise ValueError("Invalid input: text must be a non-empty string.")

            # Clean text
            processed_image = Process_image.preprocess__image(file)
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
async def predict_sentiment(request: Request, file: str = Form(...)):
    """
    POST method to handle sentiment prediction.

    Args:
        request (Request): The FastAPI request object.
        text (str): The input text to analyze.

    Returns:
        HTMLResponse: The rendered HTML response with prediction results or error messages.
    """
    # Read and encode the image as base64
    with open(file, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    # Run the prediction pipeline
    prediction_result =await pipeline_api.pipeline(file)
    print(prediction_result)
    return TEMPLATES.TemplateResponse("index.html",{"request":request,"result": prediction_result,"image_data": f"data:image/jpeg;base64,{base64_image}"})