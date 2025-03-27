from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("liver_disease_model.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        features = [float(request.form[f'feature{i}']) for i in range(1, 12)]
    except ValueError:
        return render_template("result.html", prediction="Invalid input. Please enter numeric values.")

    # Make prediction
    prediction = model.predict([features])[0]

    # Map prediction result
    result = "No Liver Disease" if prediction == 0 else "Liver Disease Detected"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
