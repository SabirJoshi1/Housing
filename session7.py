# california_price_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load Data
df = pd.read_csv("housing.csv")
df.dropna(inplace=True)

# Feature Engineering
df['AveRooms'] = df['total_rooms'] / df['households']
df['AveBedrms'] = df['total_bedrooms'] / df['households']
df['AveOccup'] = df['population'] / df['households']
df.rename(columns={
    'median_income': 'MedInc',
    'housing_median_age': 'HouseAge',
    'latitude': 'Latitude',
    'longitude': 'Longitude'
}, inplace=True)

# Select Features
features = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'population', 'AveOccup', 'Latitude', 'Longitude']]
target = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    results[name] = {
        "Train R2": r2_score(y_train, pred_train),
        "Test R2": r2_score(y_test, pred_test),
        "Test MAE": mean_absolute_error(y_test, pred_test),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, pred_test))
    }

    # Actual vs Predicted
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_train, y=pred_train, color='blue')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.title(f"{name} - Train")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_test, y=pred_test, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"{name} - Test")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.tight_layout()
    plt.show()

    # Residuals
    residuals = y_test - pred_test
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, bins=30, color='orange')
    plt.title(f"{name} - Residuals")
    plt.xlabel("Residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Model Performance
results_df = pd.DataFrame(results).T
results_df[['Test MAE', 'Test RMSE']] = results_df[['Test MAE', 'Test RMSE']].apply(pd.to_numeric)

# Error Comparison Plot
results_df[['Test MAE', 'Test RMSE']].plot(kind='bar', figsize=(10, 6), title='Model Error Comparison')
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()

print("Final Performance Metrics:\n", results_df)
