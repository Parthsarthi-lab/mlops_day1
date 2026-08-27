import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib


# Load the dataset
df = pd.read_csv("D:\\BOOKS FOR TEACHING\\MLOPS_2026_Upgrade\\mlops_day1\\data\\data.csv")

# train-test-split
X,y = df.drop(columns=["Sales"]), df["Sales"]
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=67)


# Linear Regression model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Model dump
joblib.dump(model, "models\\linear_reg_model.pkl")
