from typing import Any, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Normalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils.actions import perform_main_action
from utils.perceptions import get_perception_vector
from utils.visualization import plot_training_history


def prepare_utility_dataset(
    traces: Sequence[Sequence[Any]],
    window: int = 10,
    zero_samples: int = 10
) -> Tuple[np.ndarray, np.ndarray]:

    X, y = [], []
    for trace in traces:
        L = len(trace)
        k = min(window, L)
        n0 = min(zero_samples, L - k)

        for i in range(k):
            S = trace[-(i + 1)]
            utility = float(i + 1) / k
            X.append(S)
            y.append(utility)

        for j in range(n0):
            S = trace[-(k + j + 1)]
            X.append(S)
            y.append(0.0)

    return np.vstack(X), np.array(y, dtype=np.float32)


# def prepare_utility_dataset(traces, window=10):
#     X, y = [], []
#     for trace in traces:
#         k = min(window, len(trace))
#         for i in range(k):
#             S = trace[-(i+1)]
#             utility = float(i+1) / k
#             X.append(S)
#             y.append(utility)
#     return np.vstack(X), np.array(y, dtype=np.float32)


def train_utility_model(traces, window=10, epochs=100, save_path="utility_model.keras"):

    X, y = prepare_utility_dataset(traces, window)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    normalizer = Normalization()
    normalizer.adapt(X_train)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        normalizer,
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=8,
        callbacks=[es]
    )

    plot_training_history(history)

    y_pred = model.predict(X_test).flatten()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Utility model MAE: {mae:.4f}")
    print(f"Utility model MSE: {mse:.4f}")

    model.save(save_path)
    print(f"Guardado en {save_path}")
    return model


def novelty(candidate: np.ndarray,
            memory: np.ndarray,
            n: float = 1.0,
            m: int = 10) -> float:
    
    diffs = memory[-m:] - candidate[np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=1)
    return np.mean(dists ** n)


def intrinsic_exploration_loop(robot, sim, world_model, actions,
                                m: int = 5,
                                n: float = 1.0,
                                max_steps: int = 100,
                                goal_thresh: float = 250.0):

    # Estado inicial
    S_t = get_perception_vector(sim)
    memory = [S_t.copy()]
    log = []

    for step in range(max_steps):
        # Predicciones
        preds = []
        for a in actions:
            x = np.hstack([S_t, a/90.0]).astype(np.float32)[None,:]
            S_pred = world_model.predict(x, verbose=0)[0]
            preds.append((a, S_pred))

        # ¿Alguna predicción alcanza la meta?
        goals = [(a, S_pred) for a, S_pred in preds if S_pred[2] < goal_thresh]
        if goals:
            best_action, best_pred = min(goals, key=lambda t: t[1][1])
            print(f"Meta predicha con acción {best_action}")
            memory.append(best_pred.copy())
            S_main, ev, loc = perform_main_action(robot, sim, best_action)
            # si evadió, no revisamos meta real (se quedó en retroceso)
            if not ev and S_main[1] < goal_thresh:
                print(f"Meta real alcanzada en paso {step}")
            break

        # Novedad
        novs = [(novelty(S_pred, np.vstack(memory), n, m), a, S_pred) for a, S_pred in preds]
        novs.sort(key=lambda t: t[0], reverse=True)

        # for nov in novs:
        #     print(f"Paso {step}, Accion: {nov[1]} Novelty: {nov[0]}")

        # Top-5 intentos, con logging de pre-retroceso
        S_t1 = None
        for score, act, _ in novs[:5]:
            S_main, ev, loc = perform_main_action(robot, sim, act)
            sim.wait(0.1); robot.wait(0.1)

            if ev:
                # Logueamos la posición previa al retroceso
                log.append({
                    "step":   step,
                    "x":      loc["x"],
                    "z":      loc["z"],
                    "evaded": True
                })
                # Añadimos a memory para penalizar esa zona y seguimos
                memory.append(S_main.copy())
                continue

            # Si no evadió, aceptamos ese nuevo estado
            S_t1 = S_main
            # Podemos también loguear las transiciones exitosas:
            log.append({
                "step":   step,
                "x":      loc["x"],
                "z":      loc["z"],
                "evaded": False
            })
            break

        if S_t1 is None:
            print(f"No hay acción válida en paso {step}, dando marcha atrás.")
            robot.moveWheelsByTime(-20, -20, 1.0)
            continue

        # Actualizar memoria y estado
        memory.append(S_t1.copy())
        S_t = S_t1

        # Comprobar meta real
        if S_t[2] < goal_thresh:
            print(f"Meta real alcanzada en paso {step}")
            break
        
        if step == max_steps - 1:
            print("No se ha alcanzado la meta, no añadir este episodio a las trazas")
            memory = None
            break

    # Al final, devolvemos tanto la memoria como el log de posiciones
    return memory, log