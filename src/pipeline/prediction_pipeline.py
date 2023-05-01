import sys
import os

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            X_preprocessor_path = os.path.join('src/pipeline/artifacts', 'X_preprocessor.pkl')
            y_preprocessor_path = os.path.join('src/pipeline/artifacts', 'y_preprocessor.pkl')
            model_path = os.path.join('src/pipeline/artifacts', 'model.pkl')

            X_preprocessor = load_object(X_preprocessor_path)
            y_preprocessor = load_object(y_preprocessor_path)

            model = load_object(model_path)

            scaled_data = X_preprocessor.transform(features)
            pred = model.predict(scaled_data)

            pred = y_preprocessor.named_transformers_['output_pipeline'].inverse_transform(pred)

            return pred


        except Exception as e:
            logging.info("Exception has occured in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 x1:float,
                 x2:float,
                 x3:float,
                 x4:float,
                 x5:float,
                 x6:int,
                 x7:float,
                 x8:int):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5
        self.x6 = x6
        self.x7 = x7
        self.x8 = x8

    def get_data_as_dataframe(self):
        try:
            custom_data_dict = {
                'X1':[self.x1],
                'X2': [self.x2],
                'X3': [self.x3],
                'X4': [self.x4],
                'X5': [self.x5],
                'X6': [self.x6],
                'X7': [self.x7],
                'X8': [self.x8]
            }
            df = pd.DataFrame(custom_data_dict)
            logging.info("CustomData DataFrame created")
            return df
        except Exception as e:
            logging.info("Exception has occured in prediction pipeline (CustomData)")
            raise CustomException(e, sys)


