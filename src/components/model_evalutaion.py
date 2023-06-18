from src.logger import logging
from src.exception import CustomException
import sys
from dataclasses import dataclass

from sklearn.model_selection import KFold,cross_val_score, GridSearchCV

class ModelEvaluation:
    def __init__(self,X_train,X_test,y_train,y_test,model_dict):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_dict = model_dict
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    def initiate_model_evaluation(self):
        try:
            logging.info("Initiating model evaluation")
            scores_dict = {}

            for model_name, model in self.model_dict.items():
                model.fit(self.X_train,self.y_train)
                y_pred = model.predict(self.X_test)
                scores = cross_val_score(model, self.X_train, self.y_train, cv=self.kfold)
                mean_score = scores.mean()
                scores_dict[model_name] = mean_score

            logging.info(f"Evaluating models complete {scores_dict}")    
            return scores_dict

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_model_tuning(self,top_3models):
        try:
            logging.info("Initiating model tuning")

            logistic_regression_params = {
                'penalty': ['l2'],
                'C': [0.01, 0.1],
                'solver': ['newton-cg', 'lbfgs', 'liblinear','saga'],
                'fit_intercept': [True, False],
                'class_weight': [None, 'balanced'],
                'max_iter': [1000,1400],
                'warm_start': [True, False]
            }

            svc_params = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'degree': [2, 3, 4],
                'class_weight': [None, 'balanced']
            }

            random_forest_params = {
                'n_estimators': [300,500],
                'max_depth': [None,10],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [4,5]
            }

            gradient_boosting_params = {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.05],
                'max_depth': [5,6],
                'subsample': [0.6,0.8],
                'max_features': ['sqrt', 'log2']
            }

            xgboost_params = {
                'n_estimators': [200, 300],
                'learning_rate': [0.05, 0.01],
                'max_depth': [4, 5],
                'subsample': [0.6,0.8],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1, 0.2]
            }

            knn_params = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [20, 30, 40],
                'p': [1, 2]
            }

            param_dict = {
                            'LogisticRegression': logistic_regression_params,
                            'SVC': svc_params,
                            'RandomForestClassifier': random_forest_params,
                            'GradientBoostingClassifier': gradient_boosting_params,
                            'XGBClassifier': xgboost_params,
                            'KNeighborsClassifier': knn_params
                        }

            model_tuning_scores = {}

            for model_name in top_3models:
                params = param_dict[model_name]
                model = self.model_dict[model_name]
                grid_search = GridSearchCV(model, params, cv=self.kfold)
                grid_search.fit(self.X_train, self.y_train)
                model_tuning_scores[ model.__class__.__name__] = (grid_search.best_score_,grid_search.best_estimator_)

            logging.info(f"Model tuning done {model_tuning_scores}")
            return model_tuning_scores

        except Exception as e:
            raise CustomException(e,sys)