import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model
MODEL_PATH = "Random_Forest_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Mappings for categorical inputs to numeric values
CATEGORICAL_MAPPINGS = {
    "Make": {"Toyota": 0, "Honda": 1, "Ford": 2, "BMW": 3, "Mercedes": 4, "Audi": 5, "Other": 6},
    "Model": {"Sedan": 0, "SUV": 1, "Truck": 2, "Coupe": 3, "Van": 4, "Other": 5},
    "Fuel_Type": {"Petrol": 0, "Diesel": 1, "Hybrid": 2, "Electric": 3},
    "Transmission": {"Manual": 0, "Automatic": 1, "CVT": 2},
    "Accident_History": {"None": 0, "Minor": 1, "Major": 2},
    "Service_History": {"Full": 0, "Partial": 1, "None": 2},
    "Color": {"Black": 0, "White": 1, "Silver": 2, "Red": 3, "Blue": 4, "Other": 5},
    "Body_Type": {"Sedan": 0, "SUV": 1, "Hatchback": 2, "Truck": 3, "Convertible": 4},
    "Drivetrain": {"FWD": 0, "RWD": 1, "AWD": 2, "4WD": 3},
    "Location": {"Urban": 0, "Suburban": 1, "Rural": 2}
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded properly on the server."}), 500

    try:
        data = request.json
        
        # Extract features in exact order expected by model:
        # ['Make', 'Model', 'Year', 'Fuel_Type', 'Transmission', 'Engine_Size', 
        #  'Mileage', 'Horsepower', 'Torque', 'Owners', 'Accident_History', 
        #  'Service_History', 'Color', 'Body_Type', 'Drivetrain', 'Fuel_Efficiency', 'Location']
        
        features = [
            CATEGORICAL_MAPPINGS["Make"].get(data.get("Make"), 0),
            CATEGORICAL_MAPPINGS["Model"].get(data.get("Model"), 0),
            float(data.get("Year", 2018)),
            CATEGORICAL_MAPPINGS["Fuel_Type"].get(data.get("Fuel_Type"), 0),
            CATEGORICAL_MAPPINGS["Transmission"].get(data.get("Transmission"), 0),
            float(data.get("Engine_Size", 2.0)),
            float(data.get("Mileage", 50000)),
            float(data.get("Horsepower", 150)),
            float(data.get("Torque", 200)),
            float(data.get("Owners", 1)),
            CATEGORICAL_MAPPINGS["Accident_History"].get(data.get("Accident_History"), 0),
            CATEGORICAL_MAPPINGS["Service_History"].get(data.get("Service_History"), 0),
            CATEGORICAL_MAPPINGS["Color"].get(data.get("Color"), 0),
            CATEGORICAL_MAPPINGS["Body_Type"].get(data.get("Body_Type"), 0),
            CATEGORICAL_MAPPINGS["Drivetrain"].get(data.get("Drivetrain"), 0),
            float(data.get("Fuel_Efficiency", 15.0)),
            CATEGORICAL_MAPPINGS["Location"].get(data.get("Location"), 0)
        ]

        input_array = np.array([features])
        prediction = model.predict(input_array)[0]

        # Extract tree-level predictions for Random Forest Analytics
        tree_predictions = [tree.predict(input_array)[0] for tree in model.estimators_]
        std_dev = float(np.std(tree_predictions))
        confidence = max(0, min(100, round(100 - (std_dev / prediction * 100 if prediction else 0), 1)))

        return jsonify({
            "success": True,
            "prediction": round(float(prediction), 2),
            "confidence": confidence,
            "std_dev": round(std_dev, 2),
            "tree_predictions": [round(x, 2) for x in tree_predictions[:10]]  # Sample 10 trees
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
