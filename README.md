# Basic Flask App

This is a simple Flask web application that demonstrates a basic form submission and page navigation workflow.

---

## Features

- Runs under a Python virtual environment  
- Home page (`/`) with a form that accepts 5 user-entered values
- A Submit button that redirects the user to a results page (`/results`)  
- Results page displays a success message (no backend processing yet)

---

## Project Structure
```
flask_app/
│
├── app.py # Main Flask application
├── templates/ # HTML templates for the app
│ ├── index.html # Form input page (5 values)
│ └── results.html # Results page after form submission
└── requirements.txt # Installed dependencies
```
---
