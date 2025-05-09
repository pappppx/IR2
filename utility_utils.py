import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def prepare_utility_dataset(traces, window=10):
    X, y = [], []
    for trace in traces:
        k = min(window, len(trace))
        for i in range(k):
            S = trace[-(i+1)]      # último, penúltimo, …
            utility = float(i+1) / k
            X.append(S)
            y.append(utility)
    return np.vstack(X), np.array(y, dtype=np.float32)

def train_utility_model(traces, window=10, save_path="utility_model.keras"):
    X, y = prepare_utility_dataset(traces, window)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Sequential([
        Dense(64, activation='relu', input_shape=(6,)),
        Dense(32, activation='relu'),
        Dense(1,  activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=200,
        batch_size=32,
        callbacks=[es],
        verbose=1
    )

    # Evalúa
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    print(f"Utility model MSE: {mse:.4f}")

    model.save(save_path)
    print(f"Guardado en {save_path}")
    return model


def novelty(candidate: np.ndarray,
            memory: np.ndarray,
            n: float = 1.0) -> float:
    diffs = memory - candidate[np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=1)
    return np.mean(dists ** n)

def intrinsic_exploration_loop(robot, sim, world_model, actions,
                               n: float = 1.0,
                               max_steps: int = 100,
                               goal_thresh: float = 250.0):
    """
    Genera una traza (lista de estados 6-D) explorando por novedad.
    Cada estado S = [red_rot, red_dist, green_rot, green_dist, blue_rot, blue_dist].
    """
    from perceptions import get_simple_perceptions
    from actions     import perform_main_action

    # 1) Estado inicial
    P = get_simple_perceptions(sim)
    S_t = np.array([
        P['red_rotation'],  P['red_position'],
        P['green_rotation'],P['green_position'],
        P['blue_rotation'], P['blue_position']
    ], dtype=np.float32)
    memory = [S_t.copy()]

    for step in range(max_steps):
        # 2) Candidato para cada acción
        novs, cands = [], []
        for a in actions:
            # para predecir necesitamos [S_t, a]
            x = np.hstack([S_t, a/90.0]).astype(np.float32)[None,:]
            S_pred = world_model.predict(x)[0]  # (6,)
            cands.append((a, S_pred))
            novs.append(novelty(S_pred, np.vstack(memory), n))

        # 3) Escoger más novedoso
        best_idx   = int(np.argmax(novs))
        best_action, best_pred = cands[best_idx]

        # 4) ¿Meta predicha?
        if best_pred[1] < goal_thresh:  # usamos red_dist = índice 1
            print(f"Meta predicha con acción {best_action}")
            memory.append(best_pred.copy())
            break

        # 5) Ejecutar en robot
        S_t1 = perform_main_action(robot, sim, best_action, duration=0.5)
        sim.wait(0.1); robot.wait(0.1)
        memory.append(S_t1)
        S_t = S_t1

        # 7) ¿Meta real?
        if S_t[1] < goal_thresh:
            print(f"Meta real alcanzada en paso {step}")
            break

    return memory
