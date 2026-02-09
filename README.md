# cycle-demand-forecaster

## Bike Rental Demand Forecaster

ML2 Assignment - Tevin Heng (S10260014E)

## Project Overview
Predicts daily bike rental demand using Random Forest regression with automated quality gates.

## Repository Structure
```
cycle-demand-forecaster/
├── src/           # Preprocessing modules
├── tests/         # Quality gate tests
├── data/          # Sample dataset
├── models/        # Trained model artifacts
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Quality Gate Test
```bash
python tests/test_model.py
```

### 3. GitHub Actions
The workflow automatically runs on every push to verify model quality.

### 4. Running Streamlit 
Go to the folder which contains app.py and open terminal.
Then run  python -m streamlit run app.py

This would appear:
<img width="1908" height="685" alt="image" src="https://github.com/user-attachments/assets/aecdfb7c-5688-4c02-9a49-179e948602a9" />
Key in the values in each features to get your predicted value.

<img width="1904" height="687" alt="image" src="https://github.com/user-attachments/assets/48849f1e-c707-4b7a-ba3d-88cc0c3bce4d" />
