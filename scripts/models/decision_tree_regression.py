from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load the data
data = pd.read_csv('../../data/05-13-dataset-numerical.csv')

# Split the data into features and target
X = data.drop('spatial_reasoning', axis=1)
y = data['spatial_reasoning']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
grid = {
    'max_depth': None,
    'min_samples_split': 10,
    'min_samples_leaf': 4
}
model = DecisionTreeRegressor(**grid, random_state=42)




# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2: {r2}')

# Use case

# Load the data
use_case = pd.read_csv('../../data/use_case.csv')

# Make predictions
use_case_pred = model.predict(use_case)

print(use_case_pred)


