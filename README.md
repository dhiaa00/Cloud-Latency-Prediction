# Cloud Latency Prediction

## Overview

This project addresses the challenge of anticipating latency in cloud environments, developed as part of a datathon competition. Using machine learning techniques, specifically Random Forest Regression with hyperparameter optimization, the model predicts response times for web server requests under varying conditions.

## Methodology

### Data Preprocessing

1. Feature standardization using StandardScaler
2. Train-test split for model evaluation

### Model Development

The solution employs a Random Forest Regressor with optimized hyperparameters:

1. Initial model training with basic parameters
2. Hyperparameter optimization using GridSearchCV
3. Final model training with optimized parameters

#### Hyperparameter Grid

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

## Technical Implementation

### Key Libraries

- scikit-learn
- numpy
- pandas

### Model Training Process

```python
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter optimization
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Final model training
best_params = grid_search.best_params_
final_model = RandomForestRegressor(**best_params)
final_model.fit(X_train, y_train)
```


## Project Structure

```
cloud-latency-prediction/
├── data/
│   ├── training_data.csv    # Training data
├── notebooks/
│   └── cloud_latency_prediction.ipynb    # Main notebook with analysis
├── requirements.txt                      # Project dependencies
└── README.md                             # This documentation
```

## Setup and Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/cloud-latency-prediction.git

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebooks/cloud_latency_prediction.ipynb
```

## Future Improvements

- Experiment with other algorithms (XGBoost, LightGBM)
- Feature engineering to capture more complex patterns
- Ensemble methods combining multiple models

## Author

[Dhia eddine Elaziz]
