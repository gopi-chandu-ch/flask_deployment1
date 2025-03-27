from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
try:
    model = joblib.load("liver_disease_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevent crashing if model is missing

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure model is loaded
        if model is None:
            return render_template("result.html", prediction="Error: Model not loaded.")

        # Get data from form
        features = []
        for i in range(1, 12):
            value = request.form.get(f'feature{i}', '')
            if not value.replace(".", "", 1).isdigit():  # Allow decimals
                return render_template("result.html", prediction=f"Invalid input: {value}. Please enter numeric values.")
            features.append(float(value))

        # Ensure correct number of features
        if len(features) != 11:
            return render_template("result.html", prediction="Error: Incorrect number of features provided.")

        # Make prediction
        prediction = model.predict([features])[0]
        result = "No Liver Disease" if prediction == 0 else "Liver Disease Detected"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return render_template("result.html", prediction=f"An error occurred: {e}")

if __name__ == "__main__":
    app.run(debug=True)
