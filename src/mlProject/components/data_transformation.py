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
import numpy as np 



@dataclass
class DataTransformationConfig:
    processor_path:str = os.path.join('artifacts/models','processor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig() 
        
    def get_processor(self):
        try :
            num_cols = ['age','gender','impluse','pressurehight','pressurelow','glucose','kcm','troponin']
            cat_cols = ['class'] 
            Num_pipe = Pipeline(steps=[('imputer',SimpleImputer()),('scaler',StandardScaler())])
            processor = ColumnTransformer(transformers=[('num_pipe',Num_pipe,num_cols)])
            return processor 
         
        except Exception as e:
            logging.error('Error getting get processor file'+str(e))
            raise CustomException(e,sys)
        
    def transform(self,train_path,test_path):
        try :
            logging.info('Processer is get for data transformation')
            processor = self.get_processor()
            logging.info('Loading data from '+str(train_path)+' and '+str(test_path)) 
            # Read data from the data dir 
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            # Separate features and target  column  for train and test data  sets  and encoding categorical data  if needed  as per requirement
            train_input_cols = train_data.drop('class',axis=1)
            test_input_cols = test_data.drop('class',axis=1) 
            train_target_cols = train_data['class']
            test_target_cols = test_data['class'] 
            # Encoding categorical data
            train_target_cols = np.array(train_target_cols.map(lambda x:1 if x=='positive' else 0))
            test_target_cols = np.array(test_target_cols.map(lambda x:1 if x=='positive' else 0))  
            # Applying transformation pipeline to train and test data  sets  using fit_transform method of ColumnTransformer class  and converting the output numpy arrays back to pandas DataFrame for easier handling
            logging.info('Data transformation started')
            train_input_trans = processor.fit_transform(train_input_cols)
            test_input_trans = processor.transform(test_input_cols) 
            # Converting numpy arrays back to pandas DataFrame for easier handling
            train_data_transformed = np.concatenate((train_input_trans,train_target_cols.reshape(-1,1)),axis=1)
            test_data_transformed = np.concatenate((test_input_trans,test_target_cols.reshape(-1,1)),axis=1)
            logging.info('Data transformation completed') 
            logging.info('Saving processor started')
            save_model(processor,self.transformation_config.processor_path)
            return train_data_transformed,test_data_transformed 

        except Exception as e:
            logging.error(str(e))
            raise CustomException(e,sys)
        

        
        


