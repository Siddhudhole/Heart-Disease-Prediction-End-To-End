import os 
import sys 
import numpy as np 
from src.mlProject.exception import CustomException
from src.mlProject.logger import logging
from src.mlProject.utils import load_model
from dataclasses import dataclass
import pandas as pd 

@ dataclass
class PredictionConfig:
    model_path: str = os.path.join('artifacts/models','model.pkl')
    processor_path: str = os.path.join('artifacts/models','processor.pkl')
class Prediction():
    def __init__(self):
        self.PredictionConfig = PredictionConfig()
        self.model = load_model(self.PredictionConfig.model_path)
        self.processor = load_model(self.PredictionConfig.processor_path) 

    def predict(self,input_data:pd.DataFrame):
        try: 
            logging.info("Prediction starting")
            logging.info('Model and processor are loaded')
            input_data = self.processor.transform(input_data)  # Apply the same transformations as the training pipeline
            logging.info("input data is processed successfully{}".format(input_data))
            prediction = self.model.predict(input_data) 
            logging.info("Prediction completed successfully")  # log the success message 
            return prediction 
        except Exception as e:
            logging.error("Error in prediction"+ str(e))  # log the error
            raise CustomException(e,sys)   # re-raise the exception with additional context
        
 
        



