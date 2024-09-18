import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


df = pd.read_csv('headbrain1.csv')

# Sample data (Head Size in cm^3, Brain Weight in grams)

# Define X (features) and y (target)
X = df[['Head Size(cm^3)']]  # Independent variable
y = df['Brain Weight(grams)']  # Dependent variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values by imputing the mean
    ('scaler', StandardScaler()),                 # Feature scaling
    ('model', LinearRegression())                 # Linear Regression model
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Transform the test data and make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")