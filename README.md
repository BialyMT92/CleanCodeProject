# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project we are identifying credit card customers that are most likely to churn.

Project that was written and tested using jupiter notebook. That needs to be rewritten
in python IDE. Defined functions must works according proper sequence and give back EDA results, 
result plots and machine learning models. 

## Files and data description
Main directory:
    churn_library.py - main program
    churn_script_logging_and_tests.py - program for testing and logging functions
    requirements.txt - addition libraries that needs to be installed for python
    sequence_diag.jpeg - shows how the function runs
    ./data/"bank_data.csv" - input data
    ./images/eda - EDA result
    ./images/result - machine learning models result
    ./models - machine learning models
    ./logs/"churn_library.log" - log file after running churn_script_logging_and_tests.py

## Running Files
1. Install python using pip and additional libraries mentioned in requirements.txt.
To do it write this in to your console: "python -m pip install -r requirements.txt"
2. Run churn_library.py and wait ~15-20 minutes to crate model and results.
To do it write this in to your console: "ipython churn_library.py"
(be in directory where files are present).
3. Run churn_script_logging_and_tests.py and check ./logs/"churn_library.log"
to check if all the function runs properly.
To do it write this in to your console: "ipython churn_script_logging_and_tests.py"
(be in directory where files are present).

## Results
All the results will be stored in the main program path under ./images and ./data.
Logs after running churn_script_logging_and_tests.py will be under ./logs