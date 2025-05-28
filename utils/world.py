import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Input, Normalization, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils.visualization import plot_training_history

DEFAULT_MODEL_PATH = "models/world/"
ANGLE_COLS = [0, 1, 3, 4, 6, 7]
SIN_COLS = [0, 3, 6]
COS_COLS = [1, 4, 7]
DIST_COLS = [2, 5, 8]

def _load_and_split(csv_path, test_frac=0.2, seed=42):
    df = pd.read_csv(csv_path)
    df["action"] = df["action"] / 90.0
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    feat_cols = [
        "red_sin_t", "red_cos_t", "red_dist_t",
        "green_sin_t", "green_cos_t", "green_dist_t",
        "blue_sin_t", "blue_cos_t", "blue_dist_t",
        "action"
    ]
    targ_cols = [
        "red_sin_t1", "red_cos_t1", "red_dist_t1",
        "green_sin_t1", "green_cos_t1", "green_dist_t1",
        "blue_sin_t1", "blue_cos_t1", "blue_dist_t1"
    ]

    X = df[feat_cols].values.astype(np.float32)
    y = df[targ_cols].values.astype(np.float32)

    return train_test_split(X, y, test_size=test_frac, random_state=seed)

def _print_split_mse(y_true, y_pred, label=""):
    total_mse     = mean_squared_error(y_true, y_pred)
    mse_sin       = mean_squared_error(y_true[:, SIN_COLS],  y_pred[:, SIN_COLS])
    mse_cos       = mean_squared_error(y_true[:, COS_COLS],  y_pred[:, COS_COLS])
    mse_positions = mean_squared_error(y_true[:, DIST_COLS], y_pred[:, DIST_COLS])

    θ_true = np.arctan2(y_true[:, SIN_COLS], y_true[:, COS_COLS])
    θ_pred = np.arctan2(y_pred[:, SIN_COLS], y_pred[:, COS_COLS])

    Δθ = np.arctan2(np.sin(θ_pred - θ_true), np.cos(θ_pred - θ_true))
    Δθ_deg = Δθ * 180.0 / np.pi

    mse_angles = np.mean(Δθ_deg**2)

    print(
        f"{label} → "
        f"MSE total:        {total_mse: .4f},  "
        f"MSE ángulos (°²): {mse_angles: .4f},  "
        f"MSE senos:        {mse_sin: .4f},  "
        f"MSE cosenos:      {mse_cos: .4f},  "
        f"MSE distancias:   {mse_positions: .4f}"
    )


def _print_split_mae(y_true, y_pred, label=""):
    total_mae     = mean_absolute_error(y_true, y_pred)
    mae_sin       = mean_absolute_error(y_true[:, SIN_COLS],  y_pred[:, SIN_COLS])
    mae_cos       = mean_absolute_error(y_true[:, COS_COLS],  y_pred[:, COS_COLS])
    mae_positions = mean_absolute_error(y_true[:, DIST_COLS], y_pred[:, DIST_COLS])

    θ_true = np.arctan2(y_true[:, SIN_COLS], y_true[:, COS_COLS])
    θ_pred = np.arctan2(y_pred[:, SIN_COLS], y_pred[:, COS_COLS])

    Δθ = np.arctan2(np.sin(θ_pred - θ_true), np.cos(θ_pred - θ_true))
    Δθ_deg = np.abs(Δθ * 180.0 / np.pi)

    mae_angles = np.mean(Δθ_deg)

    print(
        f"{label} → "
        f"MAE total:       {total_mae: .4f},  "
        f"MAE ángulos (°): {mae_angles: .4f},  "
        f"MAE senos:       {mae_sin: .4f},  "
        f"MAE cosenos:     {mae_cos: .4f},  "
        f"MAE distancias:  {mae_positions: .4f}"
    )

def train_mlp_model_tf(csv_path, filename="simple_mlp_model.keras"):
    X_train, X_test, y_train, y_test = _load_and_split(csv_path)

    normalizer = Normalization()
    normalizer.adapt(X_train)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        normalizer,
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1], activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    es = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True)
    
    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-5)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=800,
        batch_size=32,
        callbacks=[es, rlr]
    )

    plot_training_history(history)

    y_pred = model.predict(X_test, verbose=0)
    
    _print_split_mse(y_test, y_pred, label="[MLP TF]")
    _print_split_mae(y_test, y_pred, label="[MLP TF]")

    model.save(DEFAULT_MODEL_PATH + filename)


def train_deep_model_tf(csv_path):
    X_train, X_test, y_train, y_test = _load_and_split(csv_path)

    normalizer = Normalization()
    normalizer.adapt(X_train)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        normalizer,
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(y_train.shape[1], activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam')
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=1500,
        batch_size=64,
        verbose=0
    )
    model.save("deep_model_profe.keras")
    y_pred = model.predict(X_test, verbose=0)
    _print_split_mse(y_test, y_pred, label="[Deep TF]")


def train_wide_and_deep_model_tf(csv_path):
    X_train, X_test, y_train, y_test = _load_and_split(csv_path)

    normalizer = Normalization()
    normalizer.adapt(X_train)

    # API funcional para Wide & Deep
    inp = Input(shape=(X_train.shape[1],))
    norm = normalizer(inp)

    # Rama wide (lineal)
    wide_out = Dense(y_train.shape[1], activation='linear')(norm)

    # Rama deep (no lineal)
    d = Dense(64, activation='relu')(norm)
    d = Dense(32, activation='relu')(d)
    deep_out = Dense(y_train.shape[1], activation='linear')(d)

    # Fusionar y proyectar
    merged = Concatenate()([wide_out, deep_out])
    out    = Dense(y_train.shape[1], activation='linear')(merged)

    model = Model(inp, out)
    model.compile(loss='mse', optimizer='adam')
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=500,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0)
    _print_split_mse(y_test, y_pred, label="[Wide&Deep TF]")