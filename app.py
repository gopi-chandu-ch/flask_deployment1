#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import joblib
import numpy as np
#import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load Trained Model
model = joblib.load('liver_disease_model.pkl')

# Encode class labels (Ensure it matches the original encoding)
class_names = ['no_disease', 'hepatitis', 'cirrhosis']  # Adjust based on your dataset

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form and convert to float
        features = [float(request.form[f'feature{i}']) for i in range(1, 12)]
    except ValueError:
        return render_template('result.html', prediction="Invalid input. Please enter numeric values.")

    # Convert to NumPy array and reshape
    features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)[0]

    # Get class name
    result = class_names[prediction]

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




