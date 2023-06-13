from src.exception import CustomException
from src.logger import logging
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import save_obj

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, lower_percentile=5, upper_percentile=95):
        self.columns = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def winsorize(self,data, lower_percentile=5, upper_percentile=95):
        lower_limit = np.percentile(data, lower_percentile)
        upper_limit = np.percentile(data, upper_percentile)
        winsorized_data = np.clip(data, lower_limit, upper_limit)
        return winsorized_data
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            for column in self.columns:
                X[column] = self.winsorize(X[column], self.lower_percentile, self.upper_percentile)
            return X
        except Exception as e:
            raise CustomException(e,sys)
        
@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self,df_path):
        self.df_path = df_path
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        
        try:
            logging.info("Creating data transformer")

            df = pd.read_csv(self.df_path)

            tar_col = ['output']
            cat_col = ['sex','cp','fbs','restecg','slp','thall','exng','caa']
            con_col = [col for col in df.columns if col not in cat_col and col not in tar_col ]

            pipe = ColumnTransformer([
                                    ('one_hot_encoder', OneHotEncoder(), cat_col),
                                    ('scaler', MinMaxScaler(), cat_col + con_col)
                            ])

            preprocessor = Pipeline([
                                            ('winsorize', WinsorizerTransformer(columns= cat_col + con_col)),
                                            ('encoder_scaler', pipe)
                                ])
            
            logging.info('Created preprocessor model')

            logging.info('Saving preprocessor model')

            save_obj(self.data_transformation_config.preprocessor_path,preprocessor)
            
            return preprocessor
         
        except Exception as  e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_set_path,test_set_path):
        
        train_df = pd.read_csv(train_set_path)
        test_df = pd.read_csv(test_set_path)

        target_column = ['output']

        logging.info("Starting data transformation")

        X_train = train_df.drop(columns=target_column)
        y_train = train_df[target_column]

        X_test = test_df.drop(columns=target_column)
        y_test = test_df[target_column]
        
        sm = SMOTE(sampling_strategy='minority',random_state=7)
        X_train, y_train = sm.fit_resample(X_train,y_train)

        preprocessor_obj = self.get_data_transformer()

        processed_input_train_arr = preprocessor_obj.fit_transform(X_train)
        processed_input_test_arr = preprocessor_obj.transform(X_test)

        train_arr = np.c_[processed_input_train_arr,np.array(y_train)]
        test_arr = np.c_[processed_input_test_arr,np.array(y_test)]

        logging.info("Data transformation completed")

        save_obj(self.data_transformation_config.preprocessor_path,preprocessor_obj)

        return (
            train_arr,
            test_arr
        )

        
