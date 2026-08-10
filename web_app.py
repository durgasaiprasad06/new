import http.server
import socketserver
import json
import os
import urllib.parse
from predict import load_champion_model, predict_loan

PORT = 8080

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LoanGuard AI - Loan Approval Prediction System</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0f1d;
            --bg-secondary: #121a2f;
            --bg-card: rgba(18, 26, 47, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --accent-amber: #f59e0b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        }

        body {
            background: radial-gradient(circle at top center, #1e293b 0%, #0a0f1d 100%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2.5rem 1.5rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Header */
        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(59, 130, 246, 0.12);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: #60a5fa;
            padding: 0.4rem 1rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        h1 {
            font-size: 2.75rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.75rem;
            letter-spacing: -0.02em;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            max-width: 600px;
            margin: 0 auto;
        }

        /* Presets Bar */
        .preset-bar {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-bottom: 2rem;
        }

        .preset-label {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 600;
        }

        .preset-btn {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 0.4rem 0.85rem;
            border-radius: 8px;
            font-size: 0.82rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .preset-btn:hover {
            background: rgba(59, 130, 246, 0.15);
            border-color: rgba(59, 130, 246, 0.4);
            color: #fff;
        }

        /* Layout Grid */
        .grid {
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 900px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }

        /* Glassmorphism Card */
        .card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.75rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .card-header h2 {
            font-size: 1.25rem;
            font-weight: 700;
        }

        /* Form Controls */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 0.4rem;
        }

        input, select {
            width: 100%;
            background: rgba(10, 15, 29, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #fff;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            font-size: 0.95rem;
            transition: all 0.2s ease;
            outline: none;
        }

        input:focus, select:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25);
        }

        select option {
            background: #121a2f;
            color: #fff;
        }

        .submit-btn {
            grid-column: span 2;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: #fff;
            border: none;
            padding: 1rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px -5px rgba(37, 99, 235, 0.4);
            margin-top: 0.5rem;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(37, 99, 235, 0.6);
        }

        /* Result Panel */
        .result-panel {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            min-height: 380px;
        }

        .status-badge {
            font-size: 1.5rem;
            font-weight: 800;
            padding: 0.5rem 1.75rem;
            border-radius: 9999px;
            margin-bottom: 1.5rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .status-approved {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-emerald);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .status-rejected {
            background: rgba(244, 63, 94, 0.15);
            color: var(--accent-rose);
            border: 1px solid rgba(244, 63, 94, 0.3);
        }

        /* Circular Gauge */
        .gauge-container {
            position: relative;
            width: 180px;
            height: 180px;
            margin-bottom: 1.5rem;
        }

        .gauge-svg {
            transform: rotate(-90deg);
        }

        .gauge-bg {
            fill: none;
            stroke: rgba(255, 255, 255, 0.05);
            stroke-width: 12;
        }

        .gauge-progress {
            fill: none;
            stroke-width: 12;
            stroke-linecap: round;
            stroke-dasharray: 440;
            stroke-dashoffset: 440;
            transition: stroke-dashoffset 1s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .gauge-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }

        .gauge-pct {
            font-size: 2.2rem;
            font-weight: 800;
        }

        .gauge-lbl {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
        }

        /* Factor Lists */
        .factors-container {
            width: 100%;
            margin-top: 1rem;
            text-align: left;
        }

        .factor-item {
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
            font-size: 0.85rem;
            padding: 0.5rem 0.75rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }

        .factor-pos {
            background: rgba(16, 185, 129, 0.08);
            color: #6ee7b7;
            border-left: 3px solid var(--accent-emerald);
        }

        .factor-neg {
            background: rgba(244, 63, 94, 0.08);
            color: #fda4af;
            border-left: 3px solid var(--accent-rose);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge">✦ Production ML Engine</div>
            <h1>Loan Approval Prediction System</h1>
            <p class="subtitle">Real-time risk scoring, feature explainability, and credit assessment powered by classical ensemble learning.</p>
        </header>

        <div class="preset-bar">
            <span class="preset-label">Test Profiles:</span>
            <button class="preset-btn" onclick="applyPreset('prime')">💎 Prime Applicant</button>
            <button class="preset-btn" onclick="applyPreset('moderate')">🏢 Moderate Risk</button>
            <button class="preset-btn" onclick="applyPreset('subprime')">⚠️ Subprime Credit</button>
            <button class="preset-btn" onclick="applyPreset('entrepreneur')">💼 Self-Employed</button>
        </div>

        <div class="grid">
            <!-- Applicant Form -->
            <div class="card">
                <div class="card-header">
                    <h2>📋 Applicant Information</h2>
                </div>
                <form id="loanForm" class="form-grid">
                    <div class="form-group">
                        <label>Gender</label>
                        <select id="Gender">
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Married Status</label>
                        <select id="Married">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Dependents</label>
                        <select id="Dependents">
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3+">3+</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Education</label>
                        <select id="Education">
                            <option value="Graduate">Graduate</option>
                            <option value="Not Graduate">Not Graduate</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Self Employed</label>
                        <select id="Self_Employed">
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Property Area</label>
                        <select id="Property_Area">
                            <option value="Semiurban">Semiurban</option>
                            <option value="Urban">Urban</option>
                            <option value="Rural">Rural</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Applicant Income ($ / month)</label>
                        <input type="number" id="ApplicantIncome" value="5000" min="0" step="100" required>
                    </div>

                    <div class="form-group">
                        <label>Co-Applicant Income ($ / month)</label>
                        <input type="number" id="CoapplicantIncome" value="2000" min="0" step="100" required>
                    </div>

                    <div class="form-group">
                        <label>Loan Amount ($ in Thousands)</label>
                        <input type="number" id="LoanAmount" value="150" min="1" step="1" required>
                    </div>

                    <div class="form-group">
                        <label>Loan Term (Months)</label>
                        <select id="Loan_Amount_Term">
                            <option value="360">360 Months (30 Yrs)</option>
                            <option value="180">180 Months (15 Yrs)</option>
                            <option value="240">240 Months (20 Yrs)</option>
                            <option value="120">120 Months (10 Yrs)</option>
                        </select>
                    </div>

                    <div class="form-group full-width">
                        <label>Credit History Record</label>
                        <select id="Credit_History">
                            <option value="1.0">1.0 - Meets Credit Guidelines (Good)</option>
                            <option value="0.0">0.0 - Has Defaults / Delinquency (Bad)</option>
                        </select>
                    </div>

                    <button type="submit" class="submit-btn">⚡ Run AI Risk Assessment</button>
                </form>
            </div>

            <!-- Real-time Results Card -->
            <div class="card">
                <div class="card-header">
                    <h2>🎯 Underwriting Verdict</h2>
                </div>
                <div id="resultPanel" class="result-panel">
                    <div id="statusBadge" class="status-badge status-approved">Approved</div>
                    
                    <div class="gauge-container">
                        <svg class="gauge-svg" width="180" height="180">
                            <circle class="gauge-bg" cx="90" cy="90" r="70"></circle>
                            <circle id="gaugeProgress" class="gauge-progress" cx="90" cy="90" r="70" stroke="#10b981"></circle>
                        </svg>
                        <div class="gauge-text">
                            <div id="probPercent" class="gauge-pct">92%</div>
                            <div class="gauge-lbl">Approval Probability</div>
                        </div>
                    </div>

                    <div id="factorsContainer" class="factors-container">
                        <!-- Populated by JavaScript -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const presets = {
            prime: {
                Gender: "Male", Married: "Yes", Dependents: "0", Education: "Graduate",
                Self_Employed: "No", Property_Area: "Semiurban", ApplicantIncome: 7500,
                CoapplicantIncome: 3500, LoanAmount: 180, Loan_Amount_Term: 360, Credit_History: "1.0"
            },
            moderate: {
                Gender: "Female", Married: "No", Dependents: "1", Education: "Graduate",
                Self_Employed: "No", Property_Area: "Urban", ApplicantIncome: 4200,
                CoapplicantIncome: 0, LoanAmount: 130, Loan_Amount_Term: 360, Credit_History: "1.0"
            },
            subprime: {
                Gender: "Male", Married: "Yes", Dependents: "2", Education: "Not Graduate",
                Self_Employed: "No", Property_Area: "Rural", ApplicantIncome: 2800,
                CoapplicantIncome: 0, LoanAmount: 200, Loan_Amount_Term: 360, Credit_History: "0.0"
            },
            entrepreneur: {
                Gender: "Female", Married: "Yes", Dependents: "0", Education: "Graduate",
                Self_Employed: "Yes", Property_Area: "Urban", ApplicantIncome: 9500,
                CoapplicantIncome: 4000, LoanAmount: 260, Loan_Amount_Term: 360, Credit_History: "1.0"
            }
        };

        function applyPreset(key) {
            const data = presets[key];
            for (const [id, val] of Object.entries(data)) {
                document.getElementById(id).value = val;
            }
            evaluateForm();
        }

        async function evaluateForm() {
            const payload = {
                Gender: document.getElementById('Gender').value,
                Married: document.getElementById('Married').value,
                Dependents: document.getElementById('Dependents').value,
                Education: document.getElementById('Education').value,
                Self_Employed: document.getElementById('Self_Employed').value,
                Property_Area: document.getElementById('Property_Area').value,
                ApplicantIncome: parseFloat(document.getElementById('ApplicantIncome').value) || 0,
                CoapplicantIncome: parseFloat(document.getElementById('CoapplicantIncome').value) || 0,
                LoanAmount: parseFloat(document.getElementById('LoanAmount').value) || 100,
                Loan_Amount_Term: parseFloat(document.getElementById('Loan_Amount_Term').value) || 360,
                Credit_History: parseFloat(document.getElementById('Credit_History').value) || 1.0
            };

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                renderResult(result);
            } catch (err) {
                console.error("API error:", err);
            }
        }

        function renderResult(res) {
            const badge = document.getElementById('statusBadge');
            const pct = document.getElementById('probPercent');
            const circle = document.getElementById('gaugeProgress');
            const factors = document.getElementById('factorsContainer');

            const isApproved = res.status === "Approved";
            badge.textContent = isApproved ? "Loan Approved" : "Loan Rejected";
            badge.className = "status-badge " + (isApproved ? "status-approved" : "status-rejected");

            const probVal = Math.round(res.probability_approved * 100);
            pct.textContent = probVal + "%";

            // Update Circle Gauge (Circumference ~ 440)
            const offset = 440 - (440 * (probVal / 100));
            circle.style.strokeDashoffset = offset;
            circle.setAttribute('stroke', isApproved ? '#10b981' : '#f43f5e');

            // Render Factor Pills
            let html = '';
            if (res.positive_factors && res.positive_factors.length > 0) {
                res.positive_factors.forEach(f => {
                    html += `<div class="factor-item factor-pos"><span>✓</span><span>${f}</span></div>`;
                });
            }
            if (res.risk_factors && res.risk_factors.length > 0) {
                res.risk_factors.forEach(f => {
                    html += `<div class="factor-item factor-neg"><span>✕</span><span>${f}</span></div>`;
                });
            }
            factors.innerHTML = html;
        }

        document.getElementById('loanForm').addEventListener('submit', (e) => {
            e.preventDefault();
            evaluateForm();
        });

        // Run initial evaluation on load
        window.addEventListener('DOMContentLoaded', () => {
            evaluateForm();
        });
    </script>
</body>
</html>
"""

class LoanPredictionHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_body.decode('utf-8'))
                result = predict_loan(data)
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=PORT):
    print(f"Starting LoanGuard AI Web Server on http://localhost:{port}")
    # Load model once at startup to warm up
    load_champion_model()
    with socketserver.TCPServer(("", port), LoanPredictionHandler) as httpd:
        print("Server running. Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    run_server()
