import os
import sys

import numpy as np
import pandas  as pd
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Exception has occured in utils.save_object")
        raise CustomException(e, sys)


def evaluate_model(y_test, y_hat):
    mae = mean_absolute_error(y_test, y_hat)
    mse = mean_squared_error(y_test, y_hat)
    rmse = np.sqrt(mse)
    r2_square = r2_score(y_test, y_hat)
    return mae, rmse, r2_square
