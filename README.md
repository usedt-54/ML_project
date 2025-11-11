# Basic Flask App

This is a simple Flask web application that demonstrates a basic form submission and page navigation workflow.

---

## Features

- Runs under a Python virtual environment  
- Home page (`/`) with a form that accepts 6 user-entered values
    Includes Total Bill, Gender, If they are a smoker, Current Day, Time, and party Size
- A Submit button that redirects the user to a results page (`/results`)  
- Results page displays The predicted tip value based on the values entered in previous screen
    Also includes R^2 and CV R^2

---

## Project Structure
```
ML_project/
│
├── app.py # Main Flask application
├── data/ # Where CSV datasets are stored
│ ├── tip.csv # Kaggle dataset
├── models/ # Where the model is stored
│ ├── model.joblib # The model
├── templates/ # HTML templates for the app
│ ├── index.html # Form input page (5 values)
│ └── results.html # Results page after form submission
├── requirements.txt # Installed dependencies
└── train_model.py # Train LinReg Model

```
---
