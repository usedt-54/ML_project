# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from joblib import dump
from pathlib import Path

CSV_PATH = "data/Student_Performance.csv"
FEATURE_COLS = ["Hours Studied","Previous Scores","Extracurricular Activities","Sleep Hours","Sample Question Papers Practiced"]
TARGET_COL = "Performance Index"
MODEL_OUT = Path("models/model.joblib")

def main():
    df = pd.read_csv(CSV_PATH)

    # Basic sanity checks
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # drop rows with NA in used columns (simple policy)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    df["Extracurricular Activities"] = (
    df["Extracurricular Activities"]
      .astype(str).str.strip().str.upper()   # normalize " y ", "Yes", etc.
      .map({"Yes": 1, "No": 0})
      .astype(float)
    )
    
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # train/test split for quick validation; adjust test_size if you want
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: scale → linear regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression())
    ])

    pipe.fit(X_train, y_train)

    # quick metrics
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Validation R^2: {r2:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    dump({"pipeline": pipe, "features": FEATURE_COLS, "target": TARGET_COL}, MODEL_OUT)
    print(f"Saved model → {MODEL_OUT.resolve()}")

if __name__ == "__main__":
    main()
