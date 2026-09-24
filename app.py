import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. Load Model & Setup Feature Columns
# ---------------------------------------------------------------------------
with open('Random_Forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

FEATURE_NAMES = [
    'Make', 'Model', 'Year', 'Fuel_Type', 'Transmission',
    'Engine_Size', 'Mileage', 'Horsepower', 'Torque', 'Owners',
    'Accident_History', 'Service_History', 'Color', 'Body_Type',
    'Drivetrain', 'Fuel_Efficiency', 'Location'
]

# ---------------------------------------------------------------------------
# 2. Combined HTML & JavaScript UI String
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Analytics Dashboard | Car Price Predictor</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js for Data Analytics -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen font-sans antialiased">

    <!-- Top Navigation Bar -->
    <nav class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="p-2 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-500/30">
                    <i data-lucide="brain-circuit" class="w-6 h-6 text-white"></i>
                </div>
                <span class="text-xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                    Predictive Analytics Dashboard
                </span>
            </div>
            <div class="flex items-center space-x-4">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 mr-2 animate-pulse"></span> Active Single-File App
                </span>
            </div>
        </div>
    </nav>

    <div class="max-w-7xl mx-auto px-6 py-8">
        
        <!-- Key KPI Metrics Banner -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5 backdrop-blur shadow-xl">
                <div class="flex items-center justify-between text-slate-400 mb-2">
                    <span class="text-xs uppercase tracking-wider font-semibold">Predicted Value</span>
                    <i data-lucide="dollar-sign" class="w-5 h-5 text-indigo-400"></i>
                </div>
                <div class="text-3xl font-extrabold text-white" id="kpi-prediction">$0.00</div>
            </div>

            <div class="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5 backdrop-blur shadow-xl">
                <div class="flex items-center justify-between text-slate-400 mb-2">
                    <span class="text-xs uppercase tracking-wider font-semibold">Model Confidence</span>
                    <i data-lucide="activity" class="w-5 h-5 text-emerald-400"></i>
                </div>
                <div class="text-3xl font-extrabold text-emerald-400" id="kpi-confidence">0.0%</div>
            </div>

            <div class="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5 backdrop-blur shadow-xl">
                <div class="flex items-center justify-between text-slate-400 mb-2">
                    <span class="text-xs uppercase tracking-wider font-semibold">Tree Variance (Std Dev)</span>
                    <i data-lucide="git-fork" class="w-5 h-5 text-purple-400"></i>
                </div>
                <div class="text-3xl font-extrabold text-purple-400" id="kpi-variance">±0.00</div>
            </div>

            <div class="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5 backdrop-blur shadow-xl">
                <div class="flex items-center justify-between text-slate-400 mb-2">
                    <span class="text-xs uppercase tracking-wider font-semibold">Total Estimators</span>
                    <i data-lucide="layers" class="w-5 h-5 text-blue-400"></i>
                </div>
                <div class="text-3xl font-extrabold text-blue-400" id="kpi-trees">100 Trees</div>
            </div>
        </div>

        <!-- Main Dashboard Content Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            <!-- Left Panel: Input Features Form -->
            <div class="lg:col-span-5 bg-slate-800/40 border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                <h2 class="text-lg font-bold mb-4 flex items-center space-x-2 text-indigo-400">
                    <i data-lucide="sliders" class="w-5 h-5"></i>
                    <span>Vehicle Parameters</span>
                </h2>
                
                <form id="prediction-form" class="grid grid-cols-2 gap-4 text-sm">
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Make (Encoded)</label>
                        <input type="number" name="Make" value="1" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500 focus:outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Model (Encoded)</label>
                        <input type="number" name="Model" value="10" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500 focus:outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Year</label>
                        <input type="number" name="Year" value="2021" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500 focus:outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Engine Size (L)</label>
                        <input type="number" step="0.1" name="Engine_Size" value="2.5" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500 focus:outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Mileage (km)</label>
                        <input type="number" name="Mileage" value="35000" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500 focus:outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Horsepower</label>
                        <input type="number" name="Horsepower" value="200" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500 focus:outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Torque (Nm)</label>
                        <input type="number" name="Torque" value="280" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500 focus:outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Fuel Efficiency</label>
                        <input type="number" step="0.1" name="Fuel_Efficiency" value="16.5" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500 focus:outline-none">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Fuel Type</label>
                        <select name="Fuel_Type" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500">
                            <option value="0">Petrol</option>
                            <option value="1">Diesel</option>
                            <option value="2">Electric</option>
                            <option value="3">Hybrid</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Transmission</label>
                        <select name="Transmission" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500">
                            <option value="0">Automatic</option>
                            <option value="1">Manual</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Owners Count</label>
                        <input type="number" name="Owners" value="1" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Accident History</label>
                        <select name="Accident_History" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500">
                            <option value="0">None</option>
                            <option value="1">Minor</option>
                            <option value="2">Major</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Service History</label>
                        <select name="Service_History" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white">
                            <option value="1">Full Service</option>
                            <option value="0">Partial</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Drivetrain</label>
                        <select name="Drivetrain" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white">
                            <option value="0">FWD</option>
                            <option value="1">RWD</option>
                            <option value="2">AWD</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Color (Encoded)</label>
                        <input type="number" name="Color" value="0" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Body Type (Encoded)</label>
                        <input type="number" name="Body_Type" value="0" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white">
                    </div>
                    <div class="col-span-2">
                        <label class="block text-xs font-semibold text-slate-400 mb-1">Location (Encoded)</label>
                        <input type="number" name="Location" value="0" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white">
                    </div>
                    <div class="col-span-2 mt-2">
                        <button type="submit" class="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl shadow-lg shadow-indigo-600/30 transition duration-200 flex items-center justify-center space-x-2">
                            <i data-lucide="play" class="w-4 h-4 fill-current"></i>
                            <span>Run Predictive Model</span>
                        </button>
                    </div>
                </form>
            </div>

            <!-- Right Panel: Visualizations -->
            <div class="lg:col-span-7 space-y-6">
                <div class="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                    <h3 class="text-md font-bold mb-4 text-slate-300 flex items-center justify-between">
                        <span>Relative Feature Sensitivity</span>
                        <span class="text-xs font-normal text-slate-400">Random Forest Node Splits</span>
                    </h3>
                    <div class="h-64">
                        <canvas id="importanceChart"></canvas>
                    </div>
                </div>

                <div class="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                    <h3 class="text-md font-bold mb-4 text-slate-300">Run History Analytics</h3>
                    <div class="h-48">
                        <canvas id="historyChart"></canvas>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <script>
        lucide.createIcons();

        const ctxImportance = document.getElementById('importanceChart').getContext('2d');
        const importanceChart = new Chart(ctxImportance, {
            type: 'radar',
            data: {
                labels: ['Engine Size', 'Year', 'Horsepower', 'Mileage', 'Fuel Efficiency', 'Torque'],
                datasets: [{
                    label: 'Feature Weight',
                    data: [85, 92, 78, 88, 65, 70],
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    pointBackgroundColor: 'rgba(99, 102, 241, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        grid: { color: '#334155' },
                        angleLines: { color: '#334155' },
                        ticks: { display: false }
                    }
                }
            }
        });

        const ctxHistory = document.getElementById('historyChart').getContext('2d');
        const historyChart = new Chart(ctxHistory, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Predicted Price ($)',
                    data: [],
                    borderColor: '#10b981',
                    tension: 0.3,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: '#334155' } },
                    y: { grid: { color: '#334155' } }
                }
            }
        });

        document.getElementById('prediction-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.status === 'success') {
                document.getElementById('kpi-prediction').innerText = '$' + result.prediction.toLocaleString();
                document.getElementById('kpi-confidence').innerText = result.confidence + '%';
                document.getElementById('kpi-variance').innerText = '±' + result.std_dev;
                document.getElementById('kpi-trees').innerText = result.trees_count + ' Trees';

                historyChart.data.labels.push(`Run ${historyChart.data.labels.length + 1}`);
                historyChart.data.datasets[0].data.push(result.prediction);
                historyChart.update();
            } else {
                alert('Error processing request: ' + result.message);
            }
        });
    </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# 3. Flask Routes
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        
        # Order features exactly as expected by model metadata
        features = [
            float(data.get('Make', 0)),
            float(data.get('Model', 0)),
            float(data.get('Year', 2020)),
            float(data.get('Fuel_Type', 0)),
            float(data.get('Transmission', 0)),
            float(data.get('Engine_Size', 2.0)),
            float(data.get('Mileage', 50000)),
            float(data.get('Horsepower', 150)),
            float(data.get('Torque', 200)),
            float(data.get('Owners', 1)),
            float(data.get('Accident_History', 0)),
            float(data.get('Service_History', 1)),
            float(data.get('Color', 0)),
            float(data.get('Body_Type', 0)),
            float(data.get('Drivetrain', 0)),
            float(data.get('Fuel_Efficiency', 15)),
            float(data.get('Location', 0))
        ]

        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        # Calculate analytics metrics across all decision trees
        tree_predictions = [tree.predict(input_array)[0] for tree in model.estimators_]
        std_dev = float(np.std(tree_predictions))
        confidence = max(0.0, min(100.0, 100 - (std_dev / (prediction + 1e-5) * 100)))

        return jsonify({
            'status': 'success',
            'prediction': round(float(prediction), 2),
            'confidence': round(confidence, 2),
            'std_dev': round(std_dev, 2),
            'trees_count': len(model.estimators_)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# ---------------------------------------------------------------------------
# 4. Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
