import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from joblib import dump
from pathlib import Path

CSV_PATH = "data/tip.csv"
FEATURE_COLS = ["total_bill", "sex", "smoker", "day", "time", "size"]
TARGET_COL = "tip"
MODEL_OUT = Path("models/model.joblib")

def main():
    df = pd.read_csv(CSV_PATH)

    # Ensure required columns exist
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Encode categoricals to numeric
    df["sex"] = (
        df["sex"].astype(str).str.strip().str.upper()
        .map({"MALE": 1.0, "M": 1.0, "FEMALE": 0.0, "F": 0.0})
    )
    df["smoker"] = (
        df["smoker"].astype(str).str.strip().str.upper()
        .map({"YES": 1.0, "Y": 1.0, "NO": 0.0, "N": 0.0})
    )
    df["time"] = (
        df["time"].astype(str).str.strip().str.upper()
        .map({"DINNER": 1.0, "LUNCH": 0.0})
    )
    # weekend flag: Saturday = 3, Sunday = 4, Thursday = 1, Friday = 2
    df["day"] = (
        df["day"].astype(str).str[:3].str.lower()
        .map({"sat": 3.0, "sun": 4.0, "thu": 1.0, "fri": 2.0})
    )

    # Validate encodings
    for col in ["sex", "smoker", "time", "day"]:
        if df[col].isna().any():
            ex = df.loc[df[col].isna(), [col]].head(5)
            raise ValueError(f"Unexpected values in '{col}'. Examples:\n{ex}")

    # Drop rows with NA in used cols
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Pipeline: scale → linear regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression())
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    pipe.fit(X_tr, y_tr)

    # Metrics
    y_pred = pipe.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    print(f"Test R^2: {r2:.4f}")

    # 5-fold CV
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    print(f"CV R^2: mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "r2_test": float(r2),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
    }
    dump({
        "pipeline": pipe,
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "metrics": metrics,
    }, "models/model.joblib")

    print(f"Saved model → {MODEL_OUT.resolve()} with features={FEATURE_COLS}")

if __name__ == "__main__":
    main()
