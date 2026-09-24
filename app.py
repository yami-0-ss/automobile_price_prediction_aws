import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

MODEL_PATH = "RandomForest_model (1).pkl"

# Load the trained RandomForest model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Mappings for categorical features expected by the model
CATEGORICAL_MAPPINGS = {
    'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'Electric': 3, 'Hybrid': 4},
    'Transmission': {'Manual': 0, 'Automatic': 1},
    'Owners': {'1st': 1, '2nd': 2, '3rd': 3, '4th+': 4},
    'Accident_History': {'No': 0, 'Yes': 1},
    'Service_History': {'Full': 0, 'Partial': 1, 'None': 2},
    'Color': {'White': 0, 'Black': 1, 'Silver': 2, 'Blue': 3, 'Red': 4, 'Other': 5},
    'Body_Type': {'Sedan': 0, 'SUV': 1, 'Hatchback': 2, 'Coupe': 3, 'Convertible': 4},
    'Drivetrain': {'FWD': 0, 'RWD': 1, 'AWD': 2, '4WD': 3},
    'Location': {'Urban': 0, 'Suburban': 1, 'Rural': 2}
}

# Embedded HTML Template (Using Raw String `r"""` to prevent invalid decimal literal string escape errors)
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vehicle Price Analytics & Valuation Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-slate-900 text-slate-100 font-sans min-h-screen">
    <nav class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <i class="fa-solid fa-chart-line text-indigo-500 text-2xl"></i>
                <span class="text-xl font-bold tracking-tight">AutoValuate <span class="text-indigo-400">Pro</span></span>
            </div>
            <span class="text-xs font-semibold px-3 py-1 bg-indigo-500/10 text-indigo-400 rounded-full border border-indigo-500/20">AWS Ready</span>
        </div>
    </nav>

    <div class="max-w-7xl mx-auto px-6 py-8">
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            <div class="lg:col-span-5 bg-slate-800/50 p-6 rounded-2xl border border-slate-700/50 shadow-xl">
                <h2 class="text-lg font-semibold text-slate-200 mb-6 flex items-center">
                    <i class="fa-solid fa-sliders text-indigo-400 mr-2"></i> Vehicle Specifications
                </h2>
                
                <form id="predictionForm" class="space-y-4">
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="text-xs text-slate-400 font-medium">Year</label>
                            <input type="number" id="Year" value="2021" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                        </div>
                        <div>
                            <label class="text-xs text-slate-400 font-medium">Fuel Type</label>
                            <select id="Fuel_Type" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                                <option>Petrol</option><option>Diesel</option><option>Electric</option><option>Hybrid</option><option>CNG</option>
                            </select>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="text-xs text-slate-400 font-medium">Engine Size (L)</label>
                            <input type="number" step="0.1" id="Engine_Size" value="2.0" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                        </div>
                        <div>
                            <label class="text-xs text-slate-400 font-medium">Transmission</label>
                            <select id="Transmission" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                                <option>Automatic</option><option>Manual</option>
                            </select>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="text-xs text-slate-400 font-medium">Mileage (km)</label>
                            <input type="number" id="Mileage" value="45000" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                        </div>
                        <div>
                            <label class="text-xs text-slate-400 font-medium">Horsepower</label>
                            <input type="number" id="Horsepower" value="180" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="text-xs text-slate-400 font-medium">Torque (Nm)</label>
                            <input type="number" id="Torque" value="250" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                        </div>
                        <div>
                            <label class="text-xs text-slate-400 font-medium">Owners</label>
                            <select id="Owners" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                                <option>1st</option><option>2nd</option><option>3rd</option><option>4th+</option>
                            </select>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="text-xs text-slate-400 font-medium">Accident History</label>
                            <select id="Accident_History" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                                <option>No</option><option>Yes</option>
                            </select>
                        </div>
                        <div>
                            <label class="text-xs text-slate-400 font-medium">Service History</label>
                            <select id="Service_History" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                                <option>Full</option><option>Partial</option><option>None</option>
                            </select>
                        </div>
                    </div>

                    <div class="grid grid-cols-3 gap-3">
                        <div>
                            <label class="text-xs text-slate-400 font-medium">Body Type</label>
                            <select id="Body_Type" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2 py-2 text-xs focus:outline-none focus:border-indigo-500">
                                <option>Sedan</option><option>SUV</option><option>Hatchback</option><option>Coupe</option>
                            </select>
                        </div>
                        <div>
                            <label class="text-xs text-slate-400 font-medium">Drivetrain</label>
                            <select id="Drivetrain" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2 py-2 text-xs focus:outline-none focus:border-indigo-500">
                                <option>FWD</option><option>AWD</option><option>RWD</option><option>4WD</option>
                            </select>
                        </div>
                        <div>
                            <label class="text-xs text-slate-400 font-medium">Location</label>
                            <select id="Location" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2 py-2 text-xs focus:outline-none focus:border-indigo-500">
                                <option>Urban</option><option>Suburban</option><option>Rural</option>
                            </select>
                        </div>
                    </div>

                    <button type="button" onclick="runPrediction()" class="w-full mt-6 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-3 rounded-lg transition-all duration-200 shadow-lg shadow-indigo-600/30 flex justify-center items-center">
                        <i class="fa-solid fa-calculator mr-2"></i> Calculate Valuation
                    </button>
                </form>
            </div>

            <div class="lg:col-span-7 space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                        <span class="text-xs text-slate-400 font-medium">Estimated Valuation</span>
                        <div class="text-2xl font-bold text-emerald-400 mt-2" id="predVal">$0.00</div>
                    </div>
                    <div class="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                        <span class="text-xs text-slate-400 font-medium">Model Variance (Std Dev)</span>
                        <div class="text-2xl font-bold text-indigo-400 mt-2" id="stdVal">$0.00</div>
                    </div>
                    <div class="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                        <span class="text-xs text-slate-400 font-medium">Estimated Range</span>
                        <div class="text-sm font-semibold text-slate-300 mt-3" id="rangeVal">$0 - $0</div>
                    </div>
                </div>

                <div class="bg-slate-800/50 p-6 rounded-2xl border border-slate-700/50 shadow-xl space-y-6">
                    <h3 class="text-sm font-semibold text-slate-300 uppercase tracking-wider">Analytics Breakdown</h3>
                    
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <h4 class="text-xs text-slate-400 mb-2">Estimator Bounds Analysis</h4>
                            <div class="h-48">
                                <canvas id="rangeChart"></canvas>
                            </div>
                        </div>
                        <div>
                            <h4 class="text-xs text-slate-400 mb-2">Key Factor Weighting (Profile)</h4>
                            <div class="h-48">
                                <canvas id="featureChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <script>
        let rangeChart, featureChart;

        function initCharts() {
            const ctxRange = document.getElementById('rangeChart').getContext('2d');
            rangeChart = new Chart(ctxRange, {
                type: 'bar',
                data: {
                    labels: ['Min Estimate', 'Random Forest Avg', 'Max Estimate'],
                    datasets: [{
                        label: 'Valuation ($)',
                        data: [0, 0, 0],
                        backgroundColor: ['#f59e0b', '#10b981', '#6366f1']
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });

            const ctxFeature = document.getElementById('featureChart').getContext('2d');
            featureChart = new Chart(ctxFeature, {
                type: 'doughnut',
                data: {
                    labels: ['Engine & Power', 'Age / Year', 'Mileage', 'Condition / History'],
                    datasets: [{
                        data: [35, 25, 25, 15],
                        backgroundColor: ['#6366f1', '#10b981', '#f59e0b', '#ec4899']
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, color: '#94a3b8' } } } }
            });
        }

        async function runPrediction() {
            const payload = {
                Year: document.getElementById('Year').value,
                Fuel_Type: document.getElementById('Fuel_Type').value,
                Engine_Size: document.getElementById('Engine_Size').value,
                Transmission: document.getElementById('Transmission').value,
                Mileage: document.getElementById('Mileage').value,
                Horsepower: document.getElementById('Horsepower').value,
                Torque: document.getElementById('Torque').value,
                Owners: document.getElementById('Owners').value,
                Accident_History: document.getElementById('Accident_History').value,
                Service_History: document.getElementById('Service_History').value,
                Body_Type: document.getElementById('Body_Type').value,
                Drivetrain: document.getElementById('Drivetrain').value,
                Location: document.getElementById('Location').value
            };

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (result.status === 'success') {
                document.getElementById('predVal').innerText = `$${result.prediction.toLocaleString()}`;
                document.getElementById('stdVal').innerText = `±$${result.std_dev.toLocaleString()}`;
                document.getElementById('rangeVal').innerText = `$${result.min_estimate.toLocaleString()} - $${result.max_estimate.toLocaleString()}`;

                rangeChart.data.datasets[0].data = [result.min_estimate, result.prediction, result.max_estimate];
                rangeChart.update();
            } else {
                alert('Prediction Error: ' + result.error);
            }
        }

        window.onload = initCharts;
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file not loaded on server.'}), 500

    try:
        data = request.json
        
        # Build 17-feature input vector matching model schema
        features = [
            float(data.get('Make', 0)),
            float(data.get('Model', 0)),
            float(data.get('Year', 2020)),
            CATEGORICAL_MAPPINGS['Fuel_Type'].get(data.get('Fuel_Type'), 0),
            CATEGORICAL_MAPPINGS['Transmission'].get(data.get('Transmission'), 0),
            float(data.get('Engine_Size', 2.0)),
            float(data.get('Mileage', 50000)),
            float(data.get('Horsepower', 150)),
            float(data.get('Torque', 200)),
            CATEGORICAL_MAPPINGS['Owners'].get(data.get('Owners'), 1),
            CATEGORICAL_MAPPINGS['Accident_History'].get(data.get('Accident_History'), 0),
            CATEGORICAL_MAPPINGS['Service_History'].get(data.get('Service_History'), 0),
            CATEGORICAL_MAPPINGS['Color'].get(data.get('Color'), 0),
            CATEGORICAL_MAPPINGS['Body_Type'].get(data.get('Body_Type'), 0),
            CATEGORICAL_MAPPINGS['Drivetrain'].get(data.get('Drivetrain'), 0),
            float(data.get('Fuel_Efficiency', 15.0)),
            CATEGORICAL_MAPPINGS['Location'].get(data.get('Location'), 0)
        ]

        input_array = np.array([features])
        prediction = model.predict(input_array)[0]

        tree_predictions = [tree.predict(input_array)[0] for tree in model.estimators_]
        std_dev = float(np.std(tree_predictions))

        return jsonify({
            'status': 'success',
            'prediction': round(float(prediction), 2),
            'std_dev': round(std_dev, 2),
            'min_estimate': round(float(np.min(tree_predictions)), 2),
            'max_estimate': round(float(np.max(tree_predictions)), 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
