import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    X_preprocessor_filepath = os.path.join('artifacts', 'X_preprocessor.pkl')
    y_preprocessor_filepath = os.path.join('artifacts', 'y_preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation initiated")

            # X_columns and y_columns
            X_columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
            y_columns = ['Y1', 'Y2']
            # preprocessor pipelines
            pipeline_X = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            pipeline_y = Pipeline(
                steps=[
                    ('y_scaler', StandardScaler())
                ]
            )
            preprocessor_X = ColumnTransformer([
                ('input_pipeline', pipeline_X, X_columns),
            ])

            preprocessor_y = ColumnTransformer([
                ('output_pipeline', pipeline_y, y_columns),
            ])

            logging.info("Creation of data transformation pipeline completed")
            return (
                preprocessor_X,
                preprocessor_y
            )

        except Exception as e:
            logging.info("Error has occured in Data Transformation stage")
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data files as pandas DataFrame
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Reading of train and test data completed.")

            # X_train, y_train, X_test, y_test split
            X_train = train_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
            X_test = test_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
            y_train, y_test = train_data[['Y1', 'Y2']], test_data[['Y1', 'Y2']]

            # Obtaining preprocessing objects
            logging.info('Obtaining preprocessing objects')
            preprocessor_X, preprocessor_y = self.get_data_transformation_obj()

            # Applying preprocessing
            logging.info("Applying preprocessing on train and test data")
            X_train = pd.DataFrame(preprocessor_X.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(preprocessor_X.transform(X_test), columns=X_train.columns)

            y_train = pd.DataFrame(preprocessor_y.fit_transform(y_train),
                                   columns=preprocessor_y.get_feature_names_out())
            y_test = pd.DataFrame(preprocessor_y.transform(y_test),
                                  columns=preprocessor_y.get_feature_names_out())
            logging.info("Preprocessing completed")

            # Concatenating X_train and y_train into a single numpy array
            train_arr = np.c_[X_train, y_train]
            # Concatenating X_test and y_test into a single numpy array
            test_arr = np.c_[X_test, y_test]

            # Saving the preprocessor objects
            save_object(file_path=self.data_transformation_config.X_preprocessor_filepath,
                        obj=preprocessor_X
                        )
            save_object(file_path=self.data_transformation_config.y_preprocessor_filepath,
                        obj=preprocessor_y
                        )
            logging.info("Preprocessor pickle files saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.X_preprocessor_filepath,
                self.data_transformation_config.y_preprocessor_filepath
            )

        except Exception as e:
            logging.info("Exception has occured in the initiate_data_transformation function")
            raise CustomException(e, sys)



