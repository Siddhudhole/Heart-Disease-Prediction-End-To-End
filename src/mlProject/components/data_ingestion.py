import os 
import sys 
from src.mlProject.logger import logging
from src.mlProject.exception import CustomException 
from dataclasses import dataclass
import pandas as pd
from src.mlProject.utils import read_sql_data 
from sklearn.model_selection import train_test_split
from pathlib import Path  

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts/data','train.csv')
    test_data_path:str = os.path.join('artifacts/data','test.csv') 
    raw_data_path:str = os.path.join('artifacts/data','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def load_data(self):
        try:
        
            logging.info("Reading data from Mysql Database")
            df = read_sql_data() 
            logging.info("Data loaded successfully from Mysql Database") 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            train_data,test_data =train_test_split(df,test_size=0.2,random_state=123)
            logging.info("Data split into train and test sets")
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            logging.info("Train data saved to artifacts/train.csv")
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Test data saved to artifacts/test.csv")
            return Path(self.ingestion_config.train_data_path),Path(self.ingestion_config.test_data_path) 
             

        except Exception as e:
            raise CustomException(e,sys) 
        



