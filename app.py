from flask import Flask, render_template, request, redirect, url_for, flash
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
METRICS = bundle.get("metrics", {})

@app.route("/", methods=["GET"])
def index():
    # Render inputs using the model's expected feature names
    return render_template("index.html", feature_names=FEATURE_COLS)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("index"))

    try:
        # Collect raw strings in the exact training order
        raw = {name: (request.form.get(name, "") or "").strip() for name in FEATURE_COLS}

        # Required fields check
        missing = [k for k, v in raw.items() if v == ""]
        if missing:
            raise ValueError(f"Missing value(s): {', '.join(missing)}")

        values = []
        for name in FEATURE_COLS:
            v = raw[name]
            lname = name.lower()
            vup = v.upper()

            if lname == "sex":
                if vup in {"MALE", "M", "1"}:
                    values.append(1.0)
                elif vup in {"FEMALE", "F", "0"}:
                    values.append(0.0)
                else:
                    raise ValueError("sex must be Male/Female (or 1/0)")
            elif lname == "smoker":
                if vup in {"YES", "Y", "1"}:
                    values.append(1.0)
                elif vup in {"NO", "N", "0"}:
                    values.append(0.0)
                else:
                    raise ValueError("smoker must be Yes/No (or 1/0)")
            elif lname == "time":
                if vup in {"DINNER", "1"}:
                    values.append(1.0)
                elif vup in {"LUNCH", "0"}:
                    values.append(0.0)
                else:
                    raise ValueError("time must be Dinner/Lunch (or 1/0)")
            elif lname == "day":
                if vup in {"THUR", "1"}:
                    values.append(1.0)
                elif vup in {"FRI", "2"}:
                    values.append(2.0)
                elif vup in {"SAT", "3"}:
                    values.append(3.0)
                elif vup in {"SUN", "4"}:
                    values.append(4.0)
                else:
                    raise ValueError("time must be Dinner/Lunch (or 1/0)")
            else:
                # numeric fields like total_bill, size
                values.append(float(v))

        # Predict with NumPy array
        X = np.array(values, dtype=float).reshape(1, -1)
        y_hat = pipeline.predict(X)[0]

        return redirect(url_for("results", pred=y_hat))

    except Exception as e:
        flash(f"Error: {e}")
        return redirect(url_for("index"))

@app.route("/results")
def results():
    pred = request.args.get("pred", type=float)
    return render_template("results.html", prediction=pred, metrics=METRICS)

if __name__ == "__main__":
    app.run(debug=True)
