import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('Salary_Data.csv')

# Check data
print("Data Preview:")
print(data.head())

# Prepare features and target
X = data[['YearsExperience']]
y = data['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ model.pkl has been created successfully.")

