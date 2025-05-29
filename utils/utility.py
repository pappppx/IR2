import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Normalization
from tensorflow.keras.callbacks import EarlyStopping
from utils.actions import perform_main_action
from utils.perceptions import get_perception_vector, get_simple_perceptions
from utils.visualization import plot_training_history

def prepare_utility_dataset(traces, window=10, decay_factor=0.8):
    X, y = [], []
    for trace in traces:
        k = min(window, len(trace))
        for i in range(k):
            S = trace[-(i + 1)]
            utility = decay_factor ** (k - i - 1)
            
            X.append(S)
            y.append(utility)
    return np.vstack(X), np.array(y, dtype=np.float32)


def train_utility_model(traces, val_split = 0.2, test_split = 0.2, window=10, epochs=300, save_path="utility_model.keras"):

    len_traces = len(traces)

    train_index = int(len_traces * (1 - val_split - test_split))
    val_index = int(len_traces * (1 - test_split))

    train_traces = traces[:train_index]
    val_traces = traces[train_index:val_index]
    test_traces = traces[val_index:]
    
    X_train, y_train = prepare_utility_dataset(train_traces, window)
    X_val, y_val = prepare_utility_dataset(val_traces, window)
    X_test, y_test   = prepare_utility_dataset(test_traces, window)

    normalizer = Normalization()
    normalizer.adapt(X_train)

    model = Sequential([
        Input(shape=(6,)),
        normalizer,
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
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

    S_t = get_perception_vector(sim)
    memory = [S_t.copy()]
    log = []

    for step in range(max_steps):

        preds = []
        for a in actions:
            x = np.hstack([S_t, a/90.0]).astype(np.float32)[None,:]
            S_pred = world_model.predict(x, verbose=0)[0]
            preds.append((a, S_pred))

        goals = [(a, S_pred) for a, S_pred in preds if S_pred[1] < goal_thresh]
        if goals:
            best_action, best_pred = min(goals, key=lambda t: t[1][1])
            print(f"Meta predicha en paso {step} con acción {best_action}")
            memory.append(best_pred.copy())
            S_main, ev, loc = perform_main_action(robot, sim, best_action)

            if not ev and S_main[1] < goal_thresh:
                print(f"Meta real alcanzada en paso {step}")
            break

        novs = [(novelty(S_pred, np.vstack(memory), n, m), a, S_pred) for a, S_pred in preds]
        novs.sort(key=lambda t: t[0], reverse=True)

        S_t1 = None
        for score, act, _ in novs[:5]:
            S_main, ev, loc = perform_main_action(robot, sim, act)
            sim.wait(0.1); robot.wait(0.1)

            if ev:
                log_position(log, step, loc, evaded=True)
                memory.append(S_main.copy())
                continue

            S_t1 = S_main
            log_position(log, step, loc, evaded=False)
            break

        if S_t1 is None:
            print(f"No hay acción válida en paso {step}, dando marcha atrás.")
            robot.moveWheelsByTime(-20, -20, 1.0)
            continue

        memory.append(S_t1.copy())
        S_t = S_t1

        if S_t[1] < goal_thresh:
            print(f"Meta real alcanzada en paso {step}")
            break
        
        if step == max_steps - 1:
            print("No se ha alcanzado la meta, no añadir este episodio a las trazas")
            memory = None
            break

    return memory, log


def log_position(log, step, loc, evaded):
    log.append({
        "step":   step,
        "x":      loc["x"],
        "z":      loc["z"],
        "evaded": evaded
    })