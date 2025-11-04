from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import numpy as np
import os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

MODEL_PATH = "models/model.joblib"

# Load model at startup
bundle = load(MODEL_PATH)
pipeline = bundle["pipeline"]
FEATURE_COLS = bundle["features"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", feature_names=FEATURE_COLS)

@app.route("/predict", methods=["POST"])
def predict():
        # pull values in the same order as FEATURE_COLS
        values = []
        for name in FEATURE_COLS:
            raw = request.form.get(name, "").strip()
            if raw == "":
                raise ValueError(f"Missing value for {name}")
            values.append(float(raw))

        X = np.array(values, dtype=float).reshape(1, -1)
        y_hat = pipeline.predict(X)[0]

        return redirect(url_for("results", pred=y_hat))
    
@app.route("/results")
def results():
    pred = request.args.get("pred", type=float)
    return render_template("results.html", prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)
