# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Data Preparation: Collect and preprocess patient data, including features like nutritional values and patient history, and label the food choices as suitable or unsuitable for diabetics.
Model Training: Split the data into training and test sets, then train a logistic regression model using the training data.
Model Evaluation: Test the trained model on the test set and evaluate its performance using metrics like accuracy, precision, and recall.
Prediction: Use the trained model to classify new food choices as suitable or unsuitable for diabetic patients.
## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: Priyadharshan S
RegisterNumber:  212223240127
*/

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,ConfusionMatrixDisplay, precision_recall_fscore_support, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
rs = 123
# Load the dataset
dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/food_items.csv"
food_df = pd.read_csv(dataset_url)
feature_cols = list(food_df.iloc[:, :-1].columns)
X_raw = food_df.iloc[:, :-1]
y_raw = food_df.iloc[:, -1:]
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)
# L2 penalty to shrink coefficients without removing any features from the model
penalty= 'l2'
# Our classification problem is multinomial
multi_class = 'multinomial'
# Use lbfgs for L2 penalty and multinomial classes
solver = 'lbfgs'
# Max iteration = 1000
max_iter = 1000
# Define a logistic regression model with above arguments
l2_model = LogisticRegression(random_state=rs, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter)
l2_model.fit(X_train, y_train)
l2_preds = l2_model.predict(X_test)
def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp)
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos
evaluate_metrics(y_test, l2_preds)

```

## Output:
![image](https://github.com/user-attachments/assets/b5ee4b19-ba57-410c-8062-91f8e733bccc)
![image](https://github.com/user-attachments/assets/477b86d1-bb36-4833-afbc-7143013e352a)
![image](https://github.com/user-attachments/assets/c3a2e35d-e990-4c7d-954c-dcd379892e18)


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
