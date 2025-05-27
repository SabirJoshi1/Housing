import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error
)

# 📥 Load and Clean Data
df = pd.read_csv("housing.csv")
df.dropna(inplace=True)

# 🛠️ Feature Engineering
df['AveRooms'] = df['total_rooms'] / df['households']
df['AveBedrms'] = df['total_bedrooms'] / df['households']
df['AveOccup'] = df['population'] / df['households']
df.rename(columns={
    'median_income': 'MedInc',
    'housing_median_age': 'HouseAge',
    'latitude': 'Latitude',
    'longitude': 'Longitude'
}, inplace=True)

# 🎯 Feature Set
features = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
               'population', 'AveOccup', 'Latitude', 'Longitude']]
target = df['median_house_value']

# 🔀 Data Splitting (80/10/10)
X_temp, X_test, y_temp, y_test = train_test_split(features, target, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, random_state=42)

# 🤖 Define Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}

# 🔁 Train & Evaluate Models on Validation Set
for name, model in models.items():
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    results[name] = {
        "Train R2": r2_score(y_train, pred_train),
        "Val R2": r2_score(y_val, pred_val),
        "Val MAE": mean_absolute_error(y_val, pred_val),
        "Val RMSE": np.sqrt(mean_squared_error(y_val, pred_val)),
        "Val MAPE (%)": mean_absolute_percentage_error(y_val, pred_val) * 100,
        "Val Median AE": median_absolute_error(y_val, pred_val)
    }

# 📊 Metrics Summary
results_df = pd.DataFrame(results).T.round(2)

# 🔥 Plot Metrics Separately
metrics_to_plot = ['Val MAE', 'Val RMSE', 'Val MAPE (%)', 'Val Median AE']
for metric in metrics_to_plot:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=results_df.index, y=results_df[metric], palette="viridis")
    plt.title(f"Validation Set - {metric}")
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# 📈 Scatter Plot (Best Model - Random Forest)
best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)
pred_val_rf = best_model.predict(X_val)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_val, y=pred_val_rf, alpha=0.6, color='green')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label='Ideal Prediction')
plt.title('Random Forest - Actual vs Predicted (Validation Set)')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 📋 Prediction Table (Top 20)
val_results = X_val.copy()
val_results['Actual Value'] = y_val.values
val_results['Predicted Value'] = pred_val_rf
val_results['Absolute Error'] = np.abs(val_results['Actual Value'] - val_results['Predicted Value'])

# Output tables
print("\n📋 Full Model Evaluation Summary:")
print(results_df)

print("\n📋 Top 20 Validation Predictions:")
print(val_results.head(20))
