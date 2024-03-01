

# Importing necessary libraries.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Loading the dataset.

dataset = pd.read_csv("ratings.csv")

# Assigning dependent and independent variables to X and y.

X = dataset.iloc[:,[0,2,3,4]].values
y = dataset.iloc[:,1].values

# Removing data with NaN values.
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")

imputer.fit(X[:,1:])
X[:,1:] = imputer.transform(X[:,1:])

dataset = dataset.drop(149, axis=0)

# Applying LabelEncoder Method

le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])

# Splitting the data into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Creating a Linear Regression object.

lr = LinearRegression()
lr.fit(X_train, y_train)

# Assigning the prediction to a variable.

y_predict = lr.predict(X_test)
