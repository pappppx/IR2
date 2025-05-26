import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Input, Normalization, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from utils.visualization import plot_training_history

# Índices en la salida de 6 columnas:
ANGLE_COLS = [0, 2, 4]  # red_rot, green_rot, blue_rot
POS_COLS   = [1, 3, 5]  # red_pos, green_pos, blue_pos

def _load_and_split(csv_path, test_frac=0.2, seed=42):
    df = pd.read_csv(csv_path)
    df["action"] = df["action"] / 90.0
    # barajar antes de split
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    feat_cols = [
        "red_rotation_t","red_position_t",
        "green_rotation_t","green_position_t",
        "blue_rotation_t","blue_position_t",
        "action"
    ]
    targ_cols = [
        "red_rotation_t1","red_position_t1",
        "green_rotation_t1","green_position_t1",
        "blue_rotation_t1","blue_position_t1"
    ]

    X = df[feat_cols].values.astype(np.float32)
    y = df[targ_cols].values.astype(np.float32)

    return train_test_split(X, y, test_size=test_frac, random_state=seed)

def _print_split_mse(y_true, y_pred, label=""):
    total_mse      = mean_squared_error(y_true, y_pred)
    mse_angles     = mean_squared_error(y_true[:, ANGLE_COLS], y_pred[:, ANGLE_COLS])
    mse_positions  = mean_squared_error(y_true[:, POS_COLS],   y_pred[:, POS_COLS])
    
    print(
        f"{label} → "
        f"MSE total:     {total_mse: .4f},  "
        f"MSE ángulos:   {mse_angles: .4f},  "
        f"MSE posiciones:{mse_positions: .4f}"
    )

def _print_split_mae(y_true, y_pred, label=""):
    total_mae      = mean_absolute_error(y_true, y_pred)
    mae_angles     = mean_absolute_error(y_true[:, ANGLE_COLS], y_pred[:, ANGLE_COLS])
    mae_positions  = mean_absolute_error(y_true[:, POS_COLS],   y_pred[:, POS_COLS])
    
    print(
        f"{label} → "
        f"MAE total:     {total_mae: .4f},  "
        f"MAE ángulos:   {mae_angles: .4f},  "
        f"MAE posiciones:{mae_positions: .4f}"
    )

def train_mlp_model_tf(csv_path):
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
    model.compile(loss='mse', optimizer='adam')
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=500,
        batch_size=32,
        callbacks=[es]
    )

    plot_training_history(history)

    y_pred = model.predict(X_test, verbose=0)
    _print_split_mse(y_test, y_pred, label="[MLP TF]")
    _print_split_mae(y_test, y_pred, label="[MLP TF]")


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