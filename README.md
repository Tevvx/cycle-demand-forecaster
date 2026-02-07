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