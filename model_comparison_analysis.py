# --- Required Libraries ---
import os
import csv
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# --- PART 1: PREPARE HIGH-DENSITY DATA FILE (Create if not exists) ---

file_name = 'drag_polar.csv'

# Check if the correct 300-point file exists, if not, create it.
# This ensures you are always running on the high-density data.
try:
    df_check = pd.read_csv(file_name)
    if len(df_check) < 290: # A simple check for the high-density file
        print("Low-density file found. Deleting and recreating with 300 points.")
        os.remove(file_name) # Remove the old file
        raise FileNotFoundError # Trigger recreation
except (FileNotFoundError, pd.errors.EmptyDataError):
    if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
        print(f"'{file_name}' not found or is empty. Creating a new high-density version (300 points)...")
        CD0, k = 0.02, 0.04
        num_points = 300
        Cl_values = np.linspace(0.2, 1.5, num_points)
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Cl', 'Cd'])
            for Cl in Cl_values:
                noise = random.uniform(-0.0005, 0.0005)
                Cd = CD0 + k * Cl**2 + noise
                writer.writerow([Cl, round(Cd, 5)])
        print(f"'{file_name}' with {num_points} data points created successfully!")

print(f"Using '{file_name}' for analysis.")
print("-" * 50)

# --- PART 2: LOAD DATA AND PREPARE VARIABLES ---

df = pd.read_csv(file_name)
X = df[['Cl']]
y = df['Cd']

X_grid_np = np.arange(X['Cl'].min(), X['Cl'].max(), 0.01).reshape(-1, 1)
X_grid_df = pd.DataFrame(X_grid_np, columns=['Cl'])


# --- PART 3: MODEL TRAINING ---

print("Training all models...")
# MODEL 1: Decision Tree
dt_model = DecisionTreeRegressor(random_state=42).fit(X, y)
y_pred_dt = dt_model.predict(X)
mse_dt, r2_dt = mean_squared_error(y, y_pred_dt), r2_score(y, y_pred_dt)

# MODEL 2: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y.values.ravel())
y_pred_rf = rf_model.predict(X)
mse_rf, r2_rf = mean_squared_error(y, y_pred_rf), r2_score(y, y_pred_rf)

# MODEL 3: Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression().fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)
mse_poly, r2_poly = mean_squared_error(y, y_pred_poly), r2_score(y, y_pred_poly)
print("All models training complete.")


# --- PART 4: FINAL METRICS COMPARISON ---

print("\n" + "-" * 65)
print("Final Performance Metrics Comparison (High-Density Data)")
print("-" * 65)
metrics_data = {
    'Metric': ['MSE (lower is better)', 'R² Score (higher is better)'],
    'Decision Tree': [f"{mse_dt:.8f}", f"{r2_dt:.8f}"],
    'Random Forest': [f"{mse_rf:.8f}", f"{r2_rf:.8f}"],
    'Polynomial Regression': [f"{mse_poly:.8f}", f"{r2_poly:.8f}"]
}
comparison_df = pd.DataFrame(metrics_data)
print(comparison_df.to_string(index=False))
print("-" * 65)

# --- PART 5: FINAL VISUALIZATION WITH CORRECTED LEGEND ---

print("\nGenerating final comparison plot with accurate R² values...")
plt.figure(figsize=(14, 9))

# Predictions for the smooth grid
y_grid_dt = dt_model.predict(X_grid_df)
y_grid_rf = rf_model.predict(X_grid_df)
y_grid_poly = poly_model.predict(poly_features.transform(X_grid_np))

# Plotting
plt.scatter(X[::10], y[::10], color='red', label='Actual Data Points (sample)', zorder=5, s=50, alpha=0.7)
# --- FIX IS HERE: Changed .4f to .6f to show more precise R² scores ---
plt.plot(X_grid_np, y_grid_dt, color='blue', linewidth=2, label=f'Decision Tree (R²: {r2_dt:.6f})')
plt.plot(X_grid_np, y_grid_rf, color='orange', linewidth=2.5, label=f'Random Forest (R²: {r2_rf:.6f})')
plt.plot(X_grid_np, y_grid_poly, color='green', linewidth=3, linestyle='--', label=f'Polynomial Regression (R²: {r2_poly:.6f})')

# Chart labels and title
plt.title('Final Model Comparison on High-Density Data (300 points)', fontsize=18)
plt.xlabel('Lift Coefficient (Cl)', fontsize=14)
plt.ylabel('Drag Coefficient (Cd)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

print("Process finished.")