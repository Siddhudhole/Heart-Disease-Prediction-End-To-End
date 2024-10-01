import os 
import sys 
from src.mlProject.logger import logging
from src.mlProject.exception import CustomException 
from dataclasses import dataclass
import pandas as pd 
from dotenv import load_dotenv
import mysql.connector 
import pickle 
from sklearn.metrics import accuracy_score , precision_score,recall_score 

load_dotenv() 
host = os.getenv('host')
user = os.getenv('user') 
password = os.getenv('password')
db = os.getenv('db') 



def read_sql_data():
    logging.info('Reading SQL database starting...')
    try :
        mydb = mysql.connector.connect(host=host,user=user,
                                password=password,db=db)
        logging.info('Connection established')
        df = pd.read_sql_query('SELECT * FROM heart_attack',mydb)
        logging.info('SQL database reading completed')
        return df 

    except Exception as e:
        logging.error(f'Error reading SQL database: {str(e)}')
        raise CustomException(e.sys) 
    
def save_model(obj,file_path):
    try :
        # Change the file permissions to read/write for the owner, group, and others
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logging.info('Model saved to '+str(file_path))
    except Exception as e:
        logging.error('Error saving model: '+str(e))
        raise CustomException(e,sys)
    
def load_model(file_path):
    try :
        model = pickle.load(open(file_path, 'rb')) 
        logging.info('Model loaded from '+str(file_path))
        return model
    except Exception as e:
        logging.error('Error loading model: '+str(e))
        raise CustomException(e,sys)
    
def evaluate_model(y_test,y_pred):
    try :
        logging.info('Evaluating model...')
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        logging.info(f'Model evaluation: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')
        return accuracy, precision, recall
    except Exception as e:
        logging.error('Error evaluating model: '+str(e))
        raise CustomException(e,sys) 


