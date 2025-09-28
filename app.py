from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and train the model
data = pd.read_csv('Salary_Data.csv')
X = data[['YearsExperience']]
y = data['Salary']

model = LinearRegression()
model.fit(X, y)

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        years_exp = float(request.form['experience'])
        model = pickle.load(open('model.pkl', 'rb'))
        prediction = model.predict(np.array([[years_exp]]))
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f"💼 Estimated Salary: ₹{output}")
    except:
        return render_template('index.html', prediction_text="❌ Invalid input. Please enter a number.")

if __name__ == "__main__":
    app.run(debug=True)
