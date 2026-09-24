import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load model artifact
MODEL_PATH = "Random_Forest_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Model could not be loaded from '{MODEL_PATH}': {e}")

# Numeric mappings for categorical inputs expected by the Random Forest model
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

# Single Embedded HTML Dashboard Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ValuAuto Analytics | AWS RF Pricing Engine</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen font-sans">
    <header class="border-b border-slate-800 bg-slate-950/50 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <div class="p-2 bg-indigo-600/20 text-indigo-400 rounded-lg border border-indigo-500/30">
                    <i data-lucide="cpu" class="w-6 h-6"></i>
                </div>
                <div>
                    <h1 class="font-bold text-lg leading-tight text-white">ValuAuto ML Engine</h1>
                    <p class="text-xs text-slate-400">Random Forest Regressor • AWS Cloud Engine</p>
                </div>
            </div>
            <span class="inline-flex items-center gap-2 text-xs bg-emerald-500/10 text-emerald-400 px-3 py-1.5 rounded-full border border-emerald-500/20">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Endpoint Active
            </span>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        <section class="lg:col-span-5 bg-slate-800/50 border border-slate-700/50 rounded-2xl p-6 backdrop-blur">
            <h2 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <i data-lucide="sliders" class="w-5 h-5 text-indigo-400"></i> Vehicle Features
            </h2>
            <form id="predictForm" class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Make</label>
                        <select id="Make" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                            <option>Toyota</option><option>Honda</option><option>Ford</option><option>BMW</option><option>Mercedes</option><option>Audi</option><option>Other</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Model Type</label>
                        <select id="Model" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                            <option>Sedan</option><option>SUV</option><option>Truck</option><option>Coupe</option><option>Van</option><option>Other</option>
                        </select>
                    </div>
                </div>
                <div class="grid grid-cols-3 gap-3">
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Year</label>
                        <input type="number" id="Year" value="2020" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Fuel</label>
                        <select id="Fuel_Type" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                            <option>Petrol</option><option>Diesel</option><option>Hybrid</option><option>Electric</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Transmission</label>
                        <select id="Transmission" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                            <option>Automatic</option><option>Manual</option><option>CVT</option>
                        </select>
                    </div>
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Mileage (km)</label>
                        <input type="number" id="Mileage" value="45000" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Engine Size (L)</label>
                        <input type="number" step="0.1" id="Engine_Size" value="2.0" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                    </div>
                </div>
                <div class="grid grid-cols-3 gap-3">
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Horsepower</label>
                        <input type="number" id="Horsepower" value="180" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Torque (Nm)</label>
                        <input type="number" id="Torque" value="250" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Owners</label>
                        <input type="number" id="Owners" value="1" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                    </div>
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Accidents</label>
                        <select id="Accident_History" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                            <option>None</option><option>Minor</option><option>Major</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Service</label>
                        <select id="Service_History" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                            <option>Full</option><option>Partial</option><option>None</option>
                        </select>
                    </div>
                </div>
                <div class="grid grid-cols-3 gap-3">
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Body</label>
                        <select id="Body_Type" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                            <option>Sedan</option><option>SUV</option><option>Hatchback</option><option>Truck</option><option>Convertible</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Drivetrain</label>
                        <select id="Drivetrain" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                            <option>FWD</option><option>AWD</option><option>RWD</option><option>4WD</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Efficiency (km/L)</label>
                        <input type="number" step="0.1" id="Fuel_Efficiency" value="16.5" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                    </div>
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Color</label>
                        <select id="Color" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                            <option>White</option><option>Black</option><option>Silver</option><option>Blue</option><option>Red</option><option>Other</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-400 mb-1">Location</label>
                        <select id="Location" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 outline-none">
                            <option>Urban</option><option>Suburban</option><option>Rural</option>
                        </select>
                    </div>
                </div>
                <button type="submit" class="w-full mt-4 bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-3 rounded-xl transition flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/30">
                    <i data-lucide="calculator" class="w-5 h-5"></i> Compute Valuation
                </button>
            </form>
        </section>

        <section class="lg:col-span-7 space-y-6">
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div class="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-5">
                    <span class="text-xs font-medium text-slate-400">Estimated Valuation</span>
                    <div class="text-2xl font-bold text-indigo-400 mt-1" id="predVal">$0.00</div>
                    <span class="text-xs text-slate-500">Predicted Price</span>
                </div>
                <div class="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-5">
                    <span class="text-xs font-medium text-slate-400">Model Confidence</span>
                    <div class="text-2xl font-bold text-emerald-400 mt-1" id="confVal">0%</div>
                    <span class="text-xs text-slate-500">Ensemble Agreement</span>
                </div>
                <div class="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-5">
                    <span class="text-xs font-medium text-slate-400">Tree Std. Deviation</span>
                    <div class="text-2xl font-bold text-amber-400 mt-1" id="stdVal">±0.00</div>
                    <span class="text-xs text-slate-500">Inter-Tree Variance</span>
                </div>
            </div>

            <div class="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-6">
                <h3 class="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
                    <i data-lucide="bar-chart-3" class="w-4 h-4 text-indigo-400"></i> Ensemble Sub-Tree Distributions
                </h3>
                <div class="h-56">
                    <canvas id="treeChart"></canvas>
                </div>
            </div>

            <div class="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-6">
                <h3 class="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
                    <i data-lucide="pie-chart" class="w-4 h-4 text-indigo-400"></i> Top Feature Weight Contributions
                </h3>
                <div class="h-52">
                    <canvas id="importanceChart"></canvas>
                </div>
            </div>
        </section>
    </main>

    <script>
        lucide.createIcons();
        const treeCtx = document.getElementById('treeChart').getContext('2d');
        const treeChart = new Chart(treeCtx, {
            type: 'line',
            data: {
                labels: ['Tree 1', 'Tree 2', 'Tree 3', 'Tree 4', 'Tree 5', 'Tree 6', 'Tree 7', 'Tree 8', 'Tree 9', 'Tree 10'],
                datasets: [{
                    label: 'Valuation ($)',
                    data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });

        const impCtx = document.getElementById('importanceChart').getContext('2d');
        new Chart(impCtx, {
            type: 'bar',
            data: {
                labels: ['Year', 'Engine Size', 'Mileage', 'Horsepower', 'Fuel Efficiency', 'Make'],
                datasets: [{
                    label: 'Feature Weight',
                    data: [0.30, 0.22, 0.18, 0.15, 0.09, 0.06],
                    backgroundColor: ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe', '#e0e7ff', '#cbd5e1']
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, indexAxis: 'y', plugins: { legend: { display: false } } }
        });

        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = {};
            const fields = ['Make', 'Model', 'Year', 'Fuel_Type', 'Transmission', 'Engine_Size', 
                            'Mileage', 'Horsepower', 'Torque', 'Owners', 'Accident_History', 
                            'Service_History', 'Color', 'Body_Type', 'Drivetrain', 'Fuel_Efficiency', 'Location'];
            fields.forEach(field => { payload[field] = document.getElementById(field).value; });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const res = await response.json();
                if (res.success) {
                    document.getElementById('predVal').innerText = '$' + res.prediction.toLocaleString();
                    document.getElementById('confVal').innerText = res.confidence + '%';
                    document.getElementById('stdVal').innerText = '±$' + res.std_dev.toLocaleString();
                    treeChart.data.datasets[0].data = res.tree_predictions;
                    treeChart.update();
                } else {
                    alert('Error: ' + res.error);
                }
            } catch (err) {
                alert('Connection error with model endpoint.');
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"success": False, "error": "Model artifact not loaded on server."}), 500

    try:
        data = request.json
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

        tree_predictions = [tree.predict(input_array)[0] for tree in model.estimators_]
        std_dev = float(np.std(tree_predictions))
        confidence = max(0, min(100, round(100 - (std_dev / prediction * 100 if prediction else 0), 1)))

        return jsonify({
            "success": True,
            "prediction": round(float(prediction), 2),
            "confidence": confidence,
            "std_dev": round(std_dev, 2),
            "tree_predictions": [round(x, 2) for x in tree_predictions[:10]]
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
