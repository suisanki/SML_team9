import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv('data_banknote_authentication.csv')

#Basically follow the instruction on page 15 of the "Day 4 Evaluation" file

# Split the data into training, evaluation (hyper-parameter + model-selection), test sets.
# Here, reconstruct the data so class 1 (fake) consist 10% of the data to replicate real world situation.
count_flag_1 = df['class'].value_counts()[1]
count_flag_0 = df['class'].value_counts()[0]
print(count_flag_1)
print(count_flag_0)

#Check the 10% value of the data
ten_percent = math.ceil((count_flag_0 * 0.1)/0.9)
df_class_1 = df[df['class'] == 1]
df_other_0 = df[df['class'] == 0]
df_class_1_reduced = df_class_1.sample(n=ten_percent, random_state=42)

#Now 10% of the whole data is class 1 (fake)
df_rebalanced = pd.concat([df_class_1_reduced, df_other_0])

X = df.drop('class', axis=1)  # Features
y = df['class']  # Target variable

#Selecting 60% for training, 20% for evaluation (hyper-parameter, model selection), 20% for test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


#Set Ensemble model (KNN+decisionTree+KernelSVC), Random Forest, and XGBoost. 
#(Reason: According to the visualized data, the data is not linear separable))

#Use Sensitivity as evaluation criteria (since the cost of false negative (classifying fake as real) is high)
#First, use GridSearchCV to find the best hyper-parameters for each model on the evaluation data

#After that, with the best hyper-parameters, evaluate the model with confidence interval using 
#crossvalidation method (bootstrapping)
# on the evaluation set.

#Calibrate the models using the best parameters found in the previous step.