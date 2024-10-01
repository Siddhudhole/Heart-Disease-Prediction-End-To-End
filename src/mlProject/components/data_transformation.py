import os,stat
import sys 
import pandas as pd 
from src.mlProject.logger import logging
from src.mlProject.exception import CustomException 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 
from src.mlProject.utils import save_model
from dataclasses import dataclass 
from pathlib import Path 
from sklearn.preprocessing import OneHotEncoder 



@dataclass
class DataTransformationConfig:
    processor_path:str = os.path.join('artifacts/models','processor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig() 
        
    def get_processor(self,data_path):
        try :
            num_cols = ['age','gender','impluse','pressurehight','pressurelow','glucose','kcm','troponin']
            cat_cols = ['class'] 
            Num_pipe = Pipeline(steps=[('imputer',SimpleImputer()),('scaler',StandardScaler())])
            Cat_pipe = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
            processor = ColumnTransformer(transformers=[('num_pipe',Num_pipe,num_cols),('cat_pipe',Cat_pipe,cat_cols)])
            return processor 
         
        except Exception as e:
            logging.error('Error getting get processor file'+str(e))
            raise CustomException(e,sys)
        
    def transform(self,train_path,test_path):
        try :
            logging.info('Processer is get for data transformation')
            processor = self.get_processor(train_path)
            logging.info('Loading data from '+str(train_path)+' and '+str(test_path)) 
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info('Data transformation started')
            train_data_transformed = processor.fit_transform(train_data)
            test_data_transformed = processor.transform(test_data) 
            logging.info('Data transformation completed') 
            logging.info('Saving processor started')
            save_model(processor,self.transformation_config.processor_path)
            return train_data_transformed,test_data_transformed 

        except Exception as e:
            logging.error(str(e))
            raise CustomException(e,sys)
        


