import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import xgboost as xgb

# --------------------------------------
# 1) LOAD DATA
# --------------------------------------
df = pd.read_csv("dataset_500.csv")

X = df[['Temperature_C', 'Voltage_V', 'Current_A']].values
Y = df.select_dtypes(include=['float64', 'int64']).drop(['Temperature_C','Voltage_V','Current_A'], axis=1).values

output_names = df.select_dtypes(include=['float64', 'int64']).drop(['Temperature_C','Voltage_V','Current_A'], axis=1).columns

# --------------------------------------
# 2) TRAIN/TEST SPLIT
# --------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "input_scaler.pkl")

# Helper dictionaries
models = {}
rmse_scores = {}
r2_scores = {}

def evaluate(model, name):
    pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(Y_test, pred))
    r2 = r2_score(Y_test, pred)
    print(f"\n{name} RESULTS:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    rmse_scores[name] = rmse
    r2_scores[name] = r2


# --------------------------------------
# 3) MODEL 1: LINEAR REGRESSION
# --------------------------------------
lin_model = MultiOutputRegressor(LinearRegression())
lin_model.fit(X_train_scaled, Y_train)
evaluate(lin_model, "Linear Regression")
joblib.dump(lin_model, "model_linear.pkl")
models["Linear Regression"] = lin_model


# --------------------------------------
# 4) MODEL 2: RANDOM FOREST
# --------------------------------------
rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42))
rf_model.fit(X_train, Y_train)
evaluate(rf_model, "Random Forest")
joblib.dump(rf_model, "model_random_forest.pkl")
models["Random Forest"] = rf_model


# --------------------------------------
# 5) MODEL 3: GRADIENT BOOSTING
# --------------------------------------
gb_model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=250, learning_rate=0.05))
gb_model.fit(X_train_scaled, Y_train)
evaluate(gb_model, "Gradient Boosting")
joblib.dump(gb_model, "model_gradient_boost.pkl")
models["Gradient Boosting"] = gb_model


# --------------------------------------
# 6) MODEL 4: XGBOOST
# --------------------------------------
xgb_model = MultiOutputRegressor(
    xgb.XGBRegressor(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror"
    )
)
xgb_model.fit(X_train_scaled, Y_train)
evaluate(xgb_model, "XGBoost")
joblib.dump(xgb_model, "model_xgboost.pkl")
models["XGBoost"] = xgb_model


# --------------------------------------
# 7) MODEL 5: NEURAL NETWORK
# --------------------------------------
nn = Sequential([
    Dense(64, activation='relu', input_shape=(3,)),
    Dense(64, activation='relu'),
    Dense(Y_train.shape[1], activation='linear')
])

nn.compile(optimizer=Adam(0.001), loss="mse")
nn.fit(X_train_scaled, Y_train, epochs=60, batch_size=16, verbose=0)

pred_nn = nn.predict(X_test_scaled)
rmse_nn = np.sqrt(mean_squared_error(Y_test, pred_nn))
r2_nn = r2_score(Y_test, pred_nn)

print("\nNEURAL NETWORK RESULTS:")
print(f"RMSE: {rmse_nn:.4f}")
print(f"R²:   {r2_nn:.4f}")

rmse_scores["Neural Network"] = rmse_nn
r2_scores["Neural Network"] = r2_nn

nn.save("model_neuralnetwork.h5")
models["Neural Network"] = nn


# --------------------------------------
# 8) FIND BEST MODEL
# --------------------------------------
best_model_name = min(rmse_scores, key=rmse_scores.get)
print(f"\n🔥 BEST MODEL: {best_model_name} (RMSE={rmse_scores[best_model_name]:.4f})")

joblib.dump(models[best_model_name], "best_model.pkl")
print("\nBest model saved as: best_model.pkl")


# ---------------------------------------------------------
# 9) METRIC PLOTS
# ---------------------------------------------------------
# RMSE plot
plt.figure(figsize=(10, 6))
plt.bar(rmse_scores.keys(), rmse_scores.values(), color='skyblue')
plt.ylabel("RMSE", fontsize=14)
plt.title("RMSE Comparison of All Models", fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
plt.savefig("rmse_plot.png")
plt.show()

# R2 plot
plt.figure(figsize=(10, 6))
plt.bar(r2_scores.keys(), r2_scores.values(), color='lightgreen')
plt.ylabel("R² Score", fontsize=14)
plt.title("R² Comparison of All Models", fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
plt.savefig("r2_plot.png")
plt.show()

# Combined Comparison Plot (RMSE & R2)
plt.figure(figsize=(12, 7))
x = np.arange(len(rmse_scores))
width = 0.35

plt.bar(x - width/2, rmse_scores.values(), width, label='RMSE', color='skyblue')
plt.bar(x + width/2, r2_scores.values(), width, label='R²', color='lightgreen')

plt.xticks(x, rmse_scores.keys(), rotation=45, fontsize=12)
plt.title("Model Performance Comparison", fontsize=18)
plt.ylabel("Score", fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("comparison_plot.png")
plt.show()
