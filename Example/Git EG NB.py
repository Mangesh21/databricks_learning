# Databricks notebook source
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Reading dataset
data = pd.read_csv('dataset.csv')

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('target_variable', axis=1), data['target_variable'], test_size=0.3)

# Training the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predicting on test data
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

