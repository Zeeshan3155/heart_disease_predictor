from src.exception import CustomException
from src.logger import logging
from src.components.model_evalutaion import ModelEvaluation
from src.utils import save_obj

import sys
import os
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_path = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Initiating model trainer")

            X_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]
            X_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]

            models_dict = {
                            'LogisticRegression': LogisticRegression(max_iter=1000),
                            'SVC': SVC(),
                            'RandomForestClassifier': RandomForestClassifier(),
                            'GradientBoostingClassifier': GradientBoostingClassifier(),
                            'XGBClassifier': XGBClassifier(),
                            'KNeighborsClassifier': KNeighborsClassifier()
                        }
            
            trainer = ModelEvaluation(X_train,X_test,y_train,y_test,models_dict)
            scores_dict = trainer.initiate_model_evaluation()
            sorted_scores = sorted(scores_dict.items(),key=lambda x:x[1],reverse=True)
            top_3models = dict(sorted_scores[:4])

            logging.info(f"Top 3 models are {top_3models}")

            tuned_model_score = trainer.initiate_model_tuning(top_3models)

            best_model = sorted(tuned_model_score.items(),key=lambda x:x[1][0])[-1]
            best_model_name = best_model[0]
            best_model = best_model[1][1]

            logging.info("Saving model")
            save_obj(self.model_path.model_path,best_model)

            logging.info(f"Model training complete, best model is {best_model_name,best_model}")
            return best_model

        except Exception as e:
            raise CustomException(e,sys)