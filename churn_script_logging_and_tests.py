"""
This function is created to test all the defined function in churn_library.py. Log will be created in the
program directory, under log file. This is not a simulator, .csv file must be present.
Testing function is done in the same order as main function is running.

Author: Mateusz Bialek
Creation date: 10/01/2023
"""

import os
import logging
import churn_library as cl
import numpy as np

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    """
    test data import
    """
    try:
        df_csv = cl.import_data(r"./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df_csv.shape[0] > 0
        assert df_csv.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df_eda):
    """
    test perform eda function
    """
    try:
        cl.perform_eda(df_eda)
        assert os.path.isfile(r"./images/eda/Churn_hist.png")
        logging.info("Churn_hist present: SUCCESS")
        assert os.path.isfile(r"./images/eda/Customer_Age_hist.png")
        logging.info("Customer_Age_hist present: SUCCESS")
        assert os.path.isfile(r"./images/eda/Martial_Status_normalize.png")
        logging.info("Martial_Status_normalize: SUCCESS")
        assert os.path.isfile(r"./images/eda/Total_Trans_Ct_histplot.png")
        logging.info("Total_Trans_Ct_histplot present: SUCCESS")
        assert os.path.isfile(r"./images/eda/HeatMap.png")
        logging.info("HeatMap present: SUCCESS")
        logging.info("Testing perform_eda: SUCCESS")
        return df_eda
    except KeyError as err1:
        logging.error("Testing perform_eda: FAILED - df does not have correct column names")
        raise err1
    except AssertionError as err2:
        logging.error("Testing perform_eda: FAILED - at least one file was not created")
        raise err2


def test_encoder_helper(df_eda, category_list, response):
    """
    test encoder helper
    """
    try:
        cl.encoder_helper(df_eda, category_list, response)

        response_lst = []
        for category in category_list:
            response_lst.append(category + '_' + response.name)

        for column in df_eda:
            if column in category_list:
                assert(np.issubdtype(df_eda[column].dtype, np.number)) == 0
            if column in response_lst:
                assert(np.issubdtype(df_eda[column].dtype, np.number)) == 1
        logging.info("Testing encoder_helper: SUCCESS - All of the columns has correct type")
        return df_eda
    except AssertionError as err:
        logging.error("Testing encoder_helper: FAILED - Columns got wrong type")


def test_perform_feature_engineering(df_ready, response):
    """
    test perform_feature_engineering
    """
    try:
        assert cl.perform_feature_engineering(df_ready, response)
        logging.info("Testing perform_feature_engineering: SUCCESS - Data are splitted to test and train")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: FAILED - Check dataFrame or response Input")


def test_train_models(X_train, X_test, y_train, y_test):
    """
    test train_models
    """
    try:
        cl.train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile(r"./models/rfc_model.pkl")
        assert os.path.isfile(r"./models/logistic_model.pkl")
        logging.info("Both models present: SUCCESS")
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: FAILED - No models detected in directory")


if __name__ == "__main__":

    test_import()

    df_csv = cl.import_data(r"./data/bank_data.csv")
    df_csv['Churn'] = df_csv['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    test_eda(df_csv)

    target_variable = df_csv['Churn']
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    test_encoder_helper(df_csv, cat_columns, target_variable)
    df_model = cl.encoder_helper(df_csv, cat_columns, target_variable)
    test_perform_feature_engineering(df_model, target_variable)

    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(df_model, target_variable)
    test_train_models(X_train, X_test, y_train, y_test)
