from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('../../data/05-13-dataset-numerical.csv')

# Split the data into features and target
X = data.drop('spatial_reasoning', axis=1)
y = data['spatial_reasoning']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# Create the model
grid = {
    'C': 10, 
    'gamma': 0.1, 
    'kernel': 'rbf'
    }

model = SVR(**grid)



# Train the model
model.fit(X_train, y_train)

mse = mean_squared_error(y_train, model.predict(X_train))
r2 = r2_score(y_train, model.predict(X_train))

print('SVR training performance:')
print(f'Mean Squared Error: {mse}')
print(f'R^2: {r2}')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('SVR test performance:')
print(f'Mean Squared Error: {mse}')
print(f'R^2: {r2}')

# Scatter plot of the predictions
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
x = np.linspace(0, 5, 100)
plt.plot(x, x, color='darkblue', linestyle='--')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('True Values vs Predictions')
plt.show()

'''
# Use case

# Load the data
use_case = pd.read_csv('../../data/use_case.csv')

# Make predictions
use_case_pred = model.predict(use_case)

print(use_case_pred)
'''