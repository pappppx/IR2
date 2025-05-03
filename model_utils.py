import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def train_simple_model(csv_path):

    df = pd.read_csv(csv_path)
    df["action"] /= 90.0
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
    
    X = df[feature_cols]
    y = df[target_cols]

    # --- 2) Split train/test 80/20 ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

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

    # --- 4) Escalado ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    # --- 5) MLP 256-256-6 ---
    n_feats = X_train_s.shape[1]
    n_tars  = y_train.shape[1]
    model = Sequential([
        Dense(64, activation='relu', input_shape=(n_feats,)),
        Dense(64, activation='relu'),
        Dense(n_tars, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    model.fit(
        X_train_s, y_train,
        validation_split=0.1,
        epochs=500,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    y_pred = model.predict(X_test_s)
    mse_mlp = mean_squared_error(y_test, y_pred)
    print(f"Trained model MSE: {mse_mlp:.4f}")
    

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
