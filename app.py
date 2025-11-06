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
    if request.method == "GET":
        return redirect(url_for("index"))

    try:
        # Map for Yes/No handling
        yn_map = {"YES": 1.0, "Y": 1.0, "NO": 0.0, "N": 0.0, "1": 1.0, "0": 0.0}

        # Use the same order your model was trained with
        feature_order = FEATURE_COLS  # typically loaded from your saved bundle

        # Collect raw strings from form
        raw = {name: request.form.get(name, "").strip() for name in feature_order}

        # Validate required fields
        missing = [k for k, v in raw.items() if v == ""]
        if missing:
            raise ValueError(f"Missing value(s): {', '.join(missing)}")

        # Convert to floats, special-casing the Yes/No field
        values = []
        for name in feature_order:
            v = raw[name]
            if name == "Extracurricular Activities":
                mapped = yn_map.get(v.upper(), None)
                if mapped is None:
                    raise ValueError("Extracurricular Activities must be Yes/No or 1/0")
                values.append(mapped)
            else:
                values.append(float(v))

        # Predict
        X = np.array(values, dtype=float).reshape(1, -1)
        y_hat = pipeline.predict(X)[0]

        return redirect(url_for("results", pred=y_hat))

    except Exception as e:
        flash(f"Error: {e}")
        return redirect(url_for("index"))
    
@app.route("/results")
def results():
    pred = request.args.get("pred", type=float)
    return render_template("results.html", prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)
