from src.utils import load_object
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig

import sys
import pandas as pd

class PredictPipeline:
    def predict(self,data):
        
        try:
            logging.info("Loading preprocessor model")
            transformer = DataTransformationConfig()
            preprocessor = load_object(transformer.preprocessor_path)
            logging.info("Loading preprocessor model complete")

            logging.info("Loading model")
            model_config = ModelTrainerConfig()
            model = load_object(model_config.model_path)
            logging.info("Model loading complete")
            
            logging.info("Transforming data")
            processed_data = preprocessor.transform(data)
            prediction = model.predict(processed_data)
            logging.info(f"Prediction complete, prediction is {prediction}")
            return prediction
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall):
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trtbps = trtbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalachh = thalachh
        self.exng = exng
        self.oldpeak = oldpeak
        self.slp = slp
        self.caa = caa
        self.thall = thall

    def create_dataframe(self):
        try:
            logging.info("Converting raw data into pandas dataframe")
            attributes = self.__dict__
            df= pd.DataFrame(attributes,index=[0])
            logging.info(df)
            return df
        
        except Exception as e:
            raise CustomException(e,sys)