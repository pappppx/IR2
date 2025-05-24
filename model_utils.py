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

def train_simple_model(csv_path):
    df = pd.read_csv(csv_path)
    df["action"] /= 90.0

    feature_cols = [
        "red_rotation_t", "red_position_t",
        "green_rotation_t", "green_position_t",
        "blue_rotation_t", "blue_position_t",
        "action"
    ]
    target_cols = [
        "red_rotation_t1","red_position_t1",
        "green_rotation_t1","green_position_t1",
        "blue_rotation_t1","blue_position_t1"
    ]
    
    X = df[feature_cols]
    y = df[target_cols]

    # 1) Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 2) Baseline: P(t+1) = P(t)
    #    Tomamos solo las 6 columnas de percepción y las convertimos a numpy
    baseline_pred = X_test[feature_cols[:-1]].values  # descartar 'action'
    y_test_arr    = y_test.values

    # MSE Ángulos y Posiciones del baseline
    base_mse_angles    = mean_squared_error(
        y_test_arr[:, ANGLE_COLS],
        baseline_pred[:, ANGLE_COLS]
    )
    base_mse_positions = mean_squared_error(
        y_test_arr[:, POS_COLS],
        baseline_pred[:, POS_COLS]
    )
    print(f"Baseline MSE (ángulos):    {base_mse_angles:.4f}")
    print(f"Baseline MSE (posiciones): {base_mse_positions:.4f}")

    # 3) Escalado
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    # 4) MLP 64-64-6 en TensorFlow/Keras
    n_feats = X_train_s.shape[1]
    n_tars  = y_train.shape[1]
    model = Sequential([
        Dense(64, activation='relu', input_shape=(n_feats,)),
        Dense(64, activation='relu'),
        Dense(n_tars, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train_s, y_train.values,
        validation_split=0.1,
        epochs=500,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    # 5) Predicción y métricas
    y_pred = model.predict(X_test_s)

    mse_total = mean_squared_error(y_test_arr, y_pred)
    mse_angles    = mean_squared_error(
        y_test_arr[:, ANGLE_COLS],
        y_pred[:,       ANGLE_COLS]
    )
    mse_positions = mean_squared_error(
        y_test_arr[:, POS_COLS],
        y_pred[:,       POS_COLS]
    )

    print(f"[Martin] -> MSE (ángulos):    {mse_angles:.4f}")
    print(f"[Martin] -> MSE (posiciones): {mse_positions:.4f}")


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
    mse_angles   = mean_squared_error(y_true[:, ANGLE_COLS], y_pred[:, ANGLE_COLS])
    mse_positions= mean_squared_error(y_true[:, POS_COLS],   y_pred[:, POS_COLS])
    print(f"{label} → MSE Ángulos: {mse_angles:.4f}, MSE Posiciones: {mse_positions:.4f}")

def train_position_model(csv_path, save_path="position_model.keras"):
    """
    Entrena un modelo que, dado:
      - red_rotation_t, red_position_t,
      - green_position_t, blue_position_t,
      - action
    predice solo las posiciones futuras:
      - red_position_t1, green_position_t1, blue_position_t1
    """
    # 1) Carga y selección de columnas
    df = pd.read_csv(csv_path)
    df["action"] = df["action"] / 90.0
    
    feature_cols = [
        "red_rotation_t",
        "red_position_t",
        "green_position_t",
        "blue_position_t",
        "action"
    ]
    target_cols = [
        "red_position_t1",
        "green_position_t1",
        "blue_position_t1"
    ]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    
    # 2) Split train/test 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3) Baseline: P_pos(t+1) = P_pos(t)
    baseline_pred = X_test[:, [1, 2, 3]]  # red_pos_t, green_pos_t, blue_pos_t
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    print(f"Baseline Position MSE: {baseline_mse:.4f}")
    
    # 4) Escalado de entradas
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    # 5) Definición de la red
    n_feats = X_train_s.shape[1]
    n_outs  = y_train.shape[1]
    model = Sequential([
        Input(shape=(n_feats,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(n_outs, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # 6) EarlyStopping
    es = EarlyStopping(monitor='val_loss', patience=300, restore_best_weights=True)
    
    # 7) Entrenamiento
    model.fit(
        X_train_s, y_train,
        validation_split=0.1,
        epochs=1500,
        batch_size=64,
        callbacks=[es],
        verbose=0
    )
    
    # 8) Evaluación
    y_pred = model.predict(X_test_s)
    pos_mse = mean_squared_error(y_test, y_pred)
    print(f"Trained Model Position MSE: {pos_mse:.4f}")
    
    # 9) Guardar modelo
    model.save(save_path)
    print(f"Modelo guardado en '{save_path}'")
    
    return model


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

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=500,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0)
    _print_split_mse(y_test, y_pred, label="[MLP TF]")


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