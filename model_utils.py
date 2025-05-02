from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import numpy as np


def train_simple_model(df_simple):

    df_simple["action"] = df_simple["action"] / 90.0

    # df_simple["red_rotation_t"] =  np.cos(df_simple["red_rotation_t"])
    # df_simple["green_rotation_t"] =  np.cos(df_simple["green_rotation_t"])
    # df_simple["blue_rotation_t"] =  np.cos(df_simple["blue_rotation_t"])

    # df_simple["red_rotation_t1"] =  np.cos(df_simple["red_rotation_t1"])
    # df_simple["green_rotation_t1"] =  np.cos(df_simple["green_rotation_t1"])
    # df_simple["blue_rotation_t1"] =  np.cos(df_simple["blue_rotation_t1"])


    feature_cols = [
        "red_rotation_t",
        "red_position_t",
        "green_rotation_t",
        "green_position_t",
        "blue_rotation_t",
        "blue_position_t",
        "action"
    ]

    target_cols = [
        "red_rotation_t1",
        "red_position_t1",
        "green_rotation_t1",
        "green_position_t1",
        "blue_rotation_t1",
        "blue_position_t1"
    ]
    
    X = df_simple[feature_cols]
    y = df_simple[target_cols]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline: assume no change in perception (P(t+1) = P(t))
    baseline_pred = X_test[[
        "red_rotation_t",
        "red_position_t",
        "green_rotation_t",
        "green_position_t",
        "blue_rotation_t",
        "blue_position_t"
    ]]
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    print(f"Baseline (P(t+1)=P(t)) MSE: {baseline_mse:.4f}")

    # MLP Regressor
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(256, 256),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        ))
    ])
    pipe.fit(X_train, y_train)
    y_pred_mlp = pipe.predict(X_test)
    mse_mlp = mean_squared_error(y_test, y_pred_mlp)
    print(f"MLP Regression MSE: {mse_mlp:.4f}")
    

def train_complex_model(df):
    # Separamos X e y
    X = df[[c for c in df.columns if not c.endswith("_t1")]]
    y = df[[c for c in df.columns if c.endswith("_t1")]]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Baseline corregido: solo sensores, no acciones ---
    sensor_cols_t = [
        c for c in X_test.columns
        if c.endswith("_t") and not c.startswith(("left_","right_"))
    ]
    baseline_pred = X_test[sensor_cols_t].copy()
    # renombrar para que coincidan con y_test
    baseline_pred.columns = [c.replace("_t", "_t1") for c in sensor_cols_t]
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    print(f"Baseline (P(t+1)=P(t)) MSE: {baseline_mse:.2f}")
    
    # --- 1) Regresión Lineal ---
    lin = MultiOutputRegressor(LinearRegression())
    lin.fit(X_train, y_train)
    y_lin = lin.predict(X_test)
    mse_lin = mean_squared_error(y_test, y_lin)
    print(f"Linear Regression MSE (complex): {mse_lin:.2f}")

    # --- 2) MLP (no lineal) ---
    mlp = MultiOutputRegressor(
        MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500, random_state=42)
    )
    mlp.fit(X_train, y_train)
    y_mlp = mlp.predict(X_test)
    mse_mlp = mean_squared_error(y_test, y_mlp)
    print(f"MLP Regression MSE (complex): {mse_mlp:.2f}")