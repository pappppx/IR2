import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Input, Normalization, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Índices en la salida de 6 columnas:
ANGLE_COLS = [0, 2, 4]  # red_rot, green_rot, blue_rot
POS_COLS   = [1, 3, 5]  # red_pos, green_pos, blue_pos


def train_mlp_model_tf(csv_path):
    print("Training MLP model...")
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
    
    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=500,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0)
    _print_split_mse(y_test, y_pred, label="[MLP TF]")


def train_deep_model_tf(csv_path):
    print("Training Deep model...")
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

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=500,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0)
    _print_split_mse(y_test, y_pred, label="[Deep TF]")


def train_wide_and_deep_model_tf(csv_path):
    print("Training Wide & Deep model...")
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

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=500,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0)
    _print_split_mse(y_test, y_pred, label="[Wide&Deep TF]")


def _load_and_split(csv_path, test_frac=0.2, seed=42):
    df = pd.read_csv(csv_path)

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
    mse_angles   = mean_squared_error(y_true[:, ANGLE_COLS], y_pred[:, ANGLE_COLS])
    mse_positions= mean_squared_error(y_true[:, POS_COLS],   y_pred[:, POS_COLS])
    print(f"{label} → MSE Ángulos: {mse_angles:.4f}, MSE Posiciones: {mse_positions:.4f}")
    

# índices para separar en y_true/y_pred
ANGLE_IDX = [0]  # angle_t1
DIST_IDX  = [1]  # dist_t1


def _load_and_split_new(csv_path, test_frac=0.2, seed=42):
    df = pd.read_csv(csv_path)
    # normalizamos la acción al rango [-1,1]
    df["action"] = df["action"] / 90.0
    
    X = df[["angle_t", "dist_t", "action"]].values.astype(np.float32)
    y = df[["angle_t1", "dist_t1"]].values.astype(np.float32)
    
    return train_test_split(X, y, test_size=test_frac, random_state=seed)


def _print_new_mse(y_true, y_pred, label=""):
    mse_total = mean_squared_error(y_true, y_pred)
    mse_angle = mean_squared_error(y_true[:, ANGLE_IDX], y_pred[:, ANGLE_IDX])
    mse_dist  = mean_squared_error(y_true[:, DIST_IDX],  y_pred[:, DIST_IDX])
    print(f"{label} → MSE total: {mse_total:.4f},  MSE ángulo: {mse_angle:.4f},  MSE distancia: {mse_dist:.4f}")


def train_mlp_model_new(csv_path):
    print("Training MLP model (new)...")
    X_train, X_test, y_train, y_test = _load_and_split_new(csv_path)
    
    normalizer = Normalization()
    normalizer.adapt(X_train)
    
    m = Sequential([
        Input(shape=(3,)),
        normalizer,
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(2,  activation="linear")
    ])
    m.compile(optimizer="adam", loss="mse")
    
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    m.fit(X_train, y_train, validation_split=0.1,
          epochs=500, batch_size=32, callbacks=[es], verbose=0)
    
    y_pred = m.predict(X_test, verbose=0)
    _print_new_mse(y_test, y_pred, label="[MLP]")


def train_deep_model_new(csv_path):
    print("Training Deep model (new)...")
    X_train, X_test, y_train, y_test = _load_and_split_new(csv_path)
    
    normalizer = Normalization()
    normalizer.adapt(X_train)
    
    m = Sequential([
        Input(shape=(3,)),
        normalizer,
        Dense(128, activation="relu"),
        Dense(64,  activation="relu"),
        Dense(32,  activation="relu"),
        Dense(2,   activation="linear")
    ])
    m.compile(optimizer="adam", loss="mse")
    
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    m.fit(X_train, y_train, validation_split=0.1,
          epochs=500, batch_size=32, callbacks=[es], verbose=0)
    
    y_pred = m.predict(X_test, verbose=0)
    _print_new_mse(y_test, y_pred, label="[Deep]")


def train_wide_and_deep_model_new(csv_path):
    print("Training Wide & Deep model (new)...")
    X_train, X_test, y_train, y_test = _load_and_split_new(csv_path)
    
    normalizer = Normalization()
    normalizer.adapt(X_train)
    
    inp  = Input(shape=(3,))
    norm = normalizer(inp)
    
    # rama wide lineal
    wide = Dense(2, activation="linear")(norm)
    
    # rama deep no lineal
    d = Dense(64, activation="relu")(norm)
    d = Dense(32, activation="relu")(d)
    deep = Dense(2, activation="linear")(d)
    
    # fusion y proyección
    merged = Concatenate()([wide, deep])
    out    = Dense(2, activation="linear")(merged)
    
    m = Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    m.fit(X_train, y_train, validation_split=0.1,
          epochs=500, batch_size=32, callbacks=[es], verbose=0)
    
    y_pred = m.predict(X_test, verbose=0)
    _print_new_mse(y_test, y_pred, label="[Wide&Deep]")