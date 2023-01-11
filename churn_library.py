"""
This function - based on csv file from the bank - create the model
that will find customers who are likely to churn.
We import the data from file, prepare dataframe, do eda and crating a model.
All calculation, grafs and results should be saved in the program path.

Author: Mateusz Bialek
Creation date: 10/01/2023
"""
import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    return df


def perform_eda(df_eda):
    """
    perform eda on df and save figures to image folder
    input:
            df_eda: pandas dataframe

    output:
            None
    """

    plt.figure(figsize=(20, 10))
    df_eda['Churn'].hist()
    plt.savefig(r"./images/eda/Churn_hist.png")

    plt.figure(figsize=(20, 10))
    df_eda['Customer_Age'].hist()
    plt.savefig(r"./images/eda/Customer_Age_hist.png")

    plt.figure(figsize=(20, 10))
    df_eda.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(r"./images/eda/Martial_Status_normalize.png")

    plt.figure(figsize=(20, 10))
    sns.histplot(df_eda['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(r"./images/eda/Total_Trans_Ct_histplot.png")

    plt.figure(figsize=(20, 10))
    sns.heatmap(df_eda.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(r"./images/eda/HeatMap.png")


def encoder_helper(df_eda, category_list, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
            naming variables or index y column]

    output:
            df_ready: pandas dataframe with new columns
    """
    # create list of columns that we want to have as an output
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio']
    response_lst = []
    for category in category_list:
        response_lst.append(category + '_' + response.name)

    merge_lst = keep_cols + response_lst

    for resp, cat in zip(response_lst, category_list):
        df_eda[resp] = df_eda.groupby(cat)[response.name].transform('mean')

    # output df with desired columns
    df_ready = df_eda[merge_lst]
    return df_ready


def perform_feature_engineering(df_ready, response):
    """
    input:
              df_ready: pandas dataframe
              response: string of response name [optional argument that could be used for
                        naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    # This cell may take up to 15-20 minutes to run
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        df_ready, response, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
            None
    """
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(r"./images/result/rf_result.png")
    plt.close('all')

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(r"./images/result/logistic_result.png")
    plt.close('all')


def feature_importance_plot(model, X_train, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_ (from file)
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_train.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_train.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_train.shape[1]), names, rotation=90)
    # save in result folder
    plt.savefig(output_pth + "feature_importances.png")
    plt.close('all')


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.close('all')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, r"./models/rfc_model.pkl")
    joblib.dump(lrc, r"./models/logistic_model.pkl")

    rfc_model = joblib.load(r"./models/rfc_model.pkl")
    lr_model = joblib.load(r"./models/logistic_model.pkl")

    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(r"./images/result/roc_curve_result.png")
    plt.close('all')

    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(r"./images/result/summary_plot.png")
    plt.close('all')

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    feature_importance_plot(cv_rfc, X_train, "./images/result/")


if __name__ == "__main__":

    # return dataframe from csv
    df_csv = import_data(r"./data/bank_data.csv")
    # create category by which we want to calculate model
    df_csv['Churn'] = df_csv['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # create target variable
    target_variable = df_csv['Churn']

    # perform eda (files saved under images/eda)
    perform_eda(df_csv)

    # create category column
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # create new dataframe prepared for creating a model
    df_model = encoder_helper(df_csv, cat_columns, target_variable)

    # create test, train frames and create model
    X_train, X_test, y_train, y_test = perform_feature_engineering(df_model, target_variable)
    train_models(X_train, X_test, y_train, y_test)
