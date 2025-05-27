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
features = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'population', 'AveOccup', 'Latitude', 'Longitude']]
target = df['median_house_value']

# 🔀 Data Splitting (80/10/10)
X_temp, X_test, y_temp, y_test = train_test_split(features, target, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, random_state=42)

# 🤖 Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}

# 🔁 Train & Evaluate on Validation Set
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

# 📊 Metrics Table
results_df = pd.DataFrame(results).T.round(2)

# 🔥 Plot Metrics Separately for Visibility
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

# 🧾 Print Full Metrics Table
print("📋 Full Evaluation Metrics on Validation Set:")
print(results_df)

# 📋 Prediction Table from Best Model (Random Forest)
best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)
pred_val_rf = best_model.predict(X_val)

val_results = X_val.copy()
val_results['Actual Value'] = y_val.values
val_results['Predicted Value'] = pred_val_rf
val_results['Absolute Error'] = np.abs(val_results['Actual Value'] - val_results['Predicted Value'])

# Show top 20 predictions
print("\n📋 Sample of Validation Prediction Table (Top 20 rows):")
print(val_results.head(20))
