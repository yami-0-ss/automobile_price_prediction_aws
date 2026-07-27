import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "RandomForest_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Warning: Model file '{MODEL_PATH}' not found in the current directory.")

# Feature options mapping based on model fields
FEATURE_OPTIONS = {
    "Make": ["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes", "Audi", "Hyundai", "Nissan", "Other"],
    "Fuel_Type": ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"],
    "Transmission": ["Manual", "Automatic", "Semi-Automatic"],
    "Accident_History": ["None", "Minor", "Major"],
    "Service_History": ["Full", "Partial", "None"],
    "Color": ["White", "Black", "Silver", "Red", "Blue", "Grey", "Other"],
    "Body_Type": ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible", "Wagon"],
    "Drivetrain": ["FWD", "RWD", "AWD", "4WD"],
    "Location": ["Urban", "Suburban", "Rural"]
}

# Embedded HTML Template with embedded CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Price Prediction</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --bg-input: #334155;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #475569;
            --success-color: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background-color: var(--bg-card);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-color);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            padding: 2.5rem 2rem;
            text-align: center;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        }

        .header p {
            color: #e2e8f0;
            font-size: 0.95rem;
            opacity: 0.9;
        }

        form {
            padding: 2rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .form-control {
            width: 100%;
            padding: 0.75rem 1rem;
            background-color: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease-in-out;
        }

        .form-control:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }

        select.form-control option {
            background-color: var(--bg-card);
            color: var(--text-main);
        }

        .actions {
            margin-top: 2rem;
            display: flex;
            justify-content: center;
        }

        .btn-submit {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
            color: white;
            font-weight: 600;
            font-size: 1rem;
            padding: 0.85rem 3rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.45);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-card {
            margin-top: 1.5rem;
            padding: 1.5rem;
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success-color);
            border-radius: 12px;
            text-align: center;
        }

        .result-card h2 {
            font-size: 1rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .result-card .price {
            font-size: 2.25rem;
            font-weight: 700;
            color: var(--success-color);
            margin-top: 0.25rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Car Price Valuation</h1>
            <p>Enter the car metrics below to estimate market value using the Random Forest model.</p>
        </div>

        <form method="POST" action="/predict">
            <div class="grid">
                <div class="form-group">
                    <label>Make</label>
                    <select name="Make" class="form-control" required>
                        {% for item in options['Make'] %}
                        <option value="{{ loop.index0 }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Model (Code/Index)</label>
                    <input type="number" name="Model" class="form-control" value="1" required>
                </div>
                <div class="form-group">
                    <label>Year</label>
                    <input type="number" name="Year" class="form-control" value="2020" min="1990" max="2026" required>
                </div>
                <div class="form-group">
                    <label>Fuel Type</label>
                    <select name="Fuel_Type" class="form-control" required>
                        {% for item in options['Fuel_Type'] %}
                        <option value="{{ loop.index0 }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Transmission</label>
                    <select name="Transmission" class="form-control" required>
                        {% for item in options['Transmission'] %}
                        <option value="{{ loop.index0 }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Engine Size (L)</label>
                    <input type="number" step="0.1" name="Engine_Size" class="form-control" value="2.0" required>
                </div>
                <div class="form-group">
                    <label>Mileage (km or miles)</label>
                    <input type="number" name="Mileage" class="form-control" value="45000" required>
                </div>
                <div class="form-group">
                    <label>Horsepower</label>
                    <input type="number" name="Horsepower" class="form-control" value="150" required>
                </div>
                <div class="form-group">
                    <label>Torque (Nm)</label>
                    <input type="number" name="Torque" class="form-control" value="200" required>
                </div>
                <div class="form-group">
                    <label>Previous Owners</label>
                    <input type="number" name="Owners" class="form-control" value="1" min="0" required>
                </div>
                <div class="form-group">
                    <label>Accident History</label>
                    <select name="Accident_History" class="form-control" required>
                        {% for item in options['Accident_History'] %}
                        <option value="{{ loop.index0 }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Service History</label>
                    <select name="Service_History" class="form-control" required>
                        {% for item in options['Service_History'] %}
                        <option value="{{ loop.index0 }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Color</label>
                    <select name="Color" class="form-control" required>
                        {% for item in options['Color'] %}
                        <option value="{{ loop.index0 }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Body Type</label>
                    <select name="Body_Type" class="form-control" required>
                        {% for item in options['Body_Type'] %}
                        <option value="{{ loop.index0 }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Drivetrain</label>
                    <select name="Drivetrain" class="form-control" required>
                        {% for item in options['Drivetrain'] %}
                        <option value="{{ loop.index0 }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label>Fuel Efficiency (km/L)</label>
                    <input type="number" step="0.1" name="Fuel_Efficiency" class="form-control" value="15.5" required>
                </div>
                <div class="form-group">
                    <label>Location</label>
                    <select name="Location" class="form-control" required>
                        {% for item in options['Location'] %}
                        <option value="{{ loop.index0 }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>

            <div class="actions">
                <button type="submit" class="btn-submit">Estimate Price</button>
            </div>

            {% if prediction is not none %}
            <div class="result-card">
                <h2>Estimated Valuation</h2>
                <div class="price">${{ "{:,.2f}".format(prediction) }}</div>
            </div>
            {% endif %}
        </form>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, options=FEATURE_OPTIONS, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model file not found. Ensure RandomForest_model.pkl is in the app root.", 500

    try:
        # Extract features according to model order
        features = [
            float(request.form.get("Make", 0)),
            float(request.form.get("Model", 0)),
            float(request.form.get("Year", 2020)),
            float(request.form.get("Fuel_Type", 0)),
            float(request.form.get("Transmission", 0)),
            float(request.form.get("Engine_Size", 2.0)),
            float(request.form.get("Mileage", 50000)),
            float(request.form.get("Horsepower", 150)),
            float(request.form.get("Torque", 200)),
            float(request.form.get("Owners", 1)),
            float(request.form.get("Accident_History", 0)),
            float(request.form.get("Service_History", 0)),
            float(request.form.get("Color", 0)),
            float(request.form.get("Body_Type", 0)),
            float(request.form.get("Drivetrain", 0)),
            float(request.form.get("Fuel_Efficiency", 15.0)),
            float(request.form.get("Location", 0))
        ]

        # Shape input for model prediction
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]

        return render_template_string(HTML_TEMPLATE, options=FEATURE_OPTIONS, prediction=prediction)
    except Exception as e:
        return f"Error making prediction: {str(e)}", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
