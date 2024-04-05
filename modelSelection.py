import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv('data_banknote_authentication.csv')

#Basically follow the instruction on page 15 of the "Day 4 Evaluation" file

# Split the data into training, evaluation (hyper-parameter), evaluation (model-selection), test sets.
# Here, reconstruct the data so class 1 (fake) consist 10% of the data to replicate real world situation.

#Set Ensemble model (KNN+decisionTree+KernelSVC), Random Forest, and XGBoost. 
#(Reason: According to the visualized data, the data is not linear separable))

#Use Sensitivity as evaluation criteria (since the cost of false negative (classifying fake as real) is high)
#First, use GridSearchCV to find the best hyper-parameters for each model on the evaluation set.

#After that, with the best hyper-parameters, evaluate the model with confidence interval using bootstrapping on the evaluation set.

#Calibrate the models using the best parameters found in the previous step.