import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model
from dataclasses import dataclass
import sys
import os


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting input and target variables")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-2],
                train_array[:, -2],
                test_array[:, :-2],
                test_array[:, -2]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'RFRegressor': RandomForestRegressor()
            }
            # dictionary for storing model reports
            model_score = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train, y_train)

                y_hat = model.predict(X_test)
                mae, rmse, r2_square = evaluate_model(y_test, y_hat)

                model_score[list(models.keys())[i]] = {'mae': mae, 'rmse': rmse, 'r2_square': r2_square}

            model_r2_score_dict = {}
            for model in model_score:
                model_r2_score_dict[model] = model_score[model]['r2_square']

            best_model_name = sorted(model_r2_score_dict, key=lambda x: x[1])[0]
            print(f'Best Model Found! , Model name: {best_model_name}, R2 Score: {model_r2_score_dict[best_model_name]}')
            print('='*40)
            logging.info(f'Best Model Found! , Model name: {best_model_name}, R2 Score: {model_r2_score_dict[best_model_name]}')

            # Saving the pickle of the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=models[best_model_name]
            )

        except Exception as e:
            logging.info("Error occured in model_trainer.ModelTrainer")
            raise CustomException(e, sys)


