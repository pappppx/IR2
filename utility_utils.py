import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from actions import perform_main_action
from perceptions import get_simple_perceptions


def prepare_utility_dataset(traces, window=10):
    X, y = [], []
    for trace in traces:
        k = min(window, len(trace))
        for i in range(k):
            S = trace[-(i+1)]
            utility = float(i+1) / k
            X.append(S)
            y.append(utility)
    return np.vstack(X), np.array(y, dtype=np.float32)


def plot_training_history(history):

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.yscale('log')
    plt.title('Model Loss Over Epochs (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def train_utility_model(traces, window=10, save_path="utility_model.keras"):

    X, y = prepare_utility_dataset(traces, window)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Sequential([
        Input(shape=(6,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=[es],
        verbose=1
    )

    plot_training_history(history)

    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    print(f"Utility model MSE: {mse:.4f}")

    model.save(save_path)
    print(f"Guardado en {save_path}")
    return model


def novelty(candidate: np.ndarray,
            memory: np.ndarray,
            n: float = 1.0) -> float:
    diffs = memory[-30:] - candidate[np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=1)
    return np.mean(dists ** n)


def intrinsic_exploration_loop(robot, sim, world_model, actions,
                                m: int = 5,
                                n: float = 1.0,
                                max_steps: int = 100,
                                goal_thresh: float = 250.0):
    """
    Variante de loop2 que, además, devuelve un `log` de posiciones
    pre-retroceso para todas las acciones que envían a go_back_if_needed.
    """
    # 1) Estado inicial
    P0 = get_simple_perceptions(sim)
    S_t = np.array([
        P0['red_rotation'],  P0['red_position'],
        P0['green_rotation'],P0['green_position'],
        P0['blue_rotation'], P0['blue_position']
    ], dtype=np.float32)
    memory = [S_t.copy()]

    # Aquí acumularemos los logs: paso, x, y, si evadió
    log = []

    for step in range(max_steps):
        # 2) Predicciones
        preds = []
        for a in actions:
            x = np.hstack([S_t, a/90.0]).astype(np.float32)[None,:]
            S_pred = world_model.predict(x, verbose=0)[0]
            preds.append((a, S_pred))

        # 3) ¿Alguna predicción alcanza la meta?
        goals = [(a, S_pred) for a, S_pred in preds if S_pred[1] < goal_thresh]
        if goals:
            best_action, best_pred = min(goals, key=lambda t: t[1][1])
            print(f"Meta predicha con acción {best_action}")
            memory.append(best_pred.copy())
            S_main, ev, loc = perform_main_action(robot, sim, best_action)
            # si evadió, no revisamos meta real (se quedó en retroceso)
            if not ev and S_main[1] < goal_thresh:
                print(f"Meta real alcanzada en paso {step}")
            break

        # 4) Novedad
        novs = [(novelty(S_pred, np.vstack(memory), n), a, S_pred)
                for a, S_pred in preds]
        novs.sort(key=lambda t: t[0], reverse=True)

        # 5) Top-5 intentos, con logging de pre-retroceso
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

        # 6) Actualizar memoria y estado
        memory.append(S_t1.copy())
        S_t = S_t1

        # 7) Comprobar meta real
        if S_t[1] < goal_thresh:
            print(f"Meta real alcanzada en paso {step}")
            break
        
        if step == max_steps - 1:
            print("No se ha alcanzado la meta, no añadir este episodio a las trazas")
            memory = None
            break

    # Al final, devolvemos tanto la memoria como el log de posiciones
    return memory, log