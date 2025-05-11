import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from perceptions import get_simple_perceptions
from actions import perform_main_action
from obstacle_avoidance import undo_if_needed

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


def novelty(candidate: np.ndarray, memory: np.ndarray, n: float = 1.0) -> float:

    diffs = memory - candidate[np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=1)
    return np.mean(dists ** n)


# def intrinsic_exploration_loop(robot, sim, world_model, actions,
#                                n: float = 1.0,
#                                max_steps: int = 100,
#                                goal_thresh: float = 250.0):
#     """
#     Genera una traza (lista de estados 6-D) explorando por novedad.
#     Cada estado S = [red_rot, red_dist, green_rot, green_dist, blue_rot, blue_dist].
#     """

#     # 1) Estado inicial
#     P = get_simple_perceptions(sim)
#     S_t = np.array([
#         P['red_rotation'],  P['red_position'],
#         P['green_rotation'],P['green_position'],
#         P['blue_rotation'], P['blue_position']
#     ], dtype=np.float32)
#     memory = [S_t.copy()]

#     for step in range(max_steps):
#         print(f"\nEpoch {step+1}/{max_steps}:")

#         # 2) Candidato para cada acción
#         novelties, candidates = [], []
#         for a in actions:
#             x = np.hstack([S_t, a/90.0]).astype(np.float32)[None,:]
#             S_pred = world_model.predict(x, verbose=0)[0]
#             candidates.append((a, S_pred))
#             nov = novelty(S_pred, np.vstack(memory), n)
#             novelties.append(nov)
#             print(nov)

#         # 3) Escoger más novedoso
#         best_idx = int(np.argmax(novelties))
#         best_action, best_pred = candidates[best_idx]

#         # 4) ¿Meta predicha?
#         if best_pred[1] < goal_thresh:  # usamos red_dist = índice 1
#             print(f"Meta predicha con acción {best_action}")
#             memory.append(best_pred.copy())
#             break

#         # 5) Ejecutar en robot
#         S_t1 = perform_main_action(robot, sim, best_action, duration=0.5)
#         if not undo_if_needed(robot, best_action): 
#             memory.append(S_t1)
#             S_t = S_t1

#         # 7) ¿Meta real?
#         if S_t[1] < goal_thresh:
#             print(f"Meta real alcanzada en paso {step}")
#             break

#     return memory

def intrinsic_exploration_loop(robot, sim, world_model, actions,
                               n: float = 1.0,
                               max_steps: int = 100,
                               goal_thresh: float = 250.0):
    """
    Genera una traza (lista de estados 6-D) explorando por novedad.
    Cada estado S = [red_rot, red_dist, green_rot, green_dist, blue_rot, blue_dist].
    """

    # Estado inicial
    P = get_simple_perceptions(sim)
    S_t = np.array([
        P['red_rotation'],  P['red_position'],
        P['green_rotation'],P['green_position'],
        P['blue_rotation'], P['blue_position']
    ], dtype=np.float32)
    memory = [S_t.copy()]

    for step in range(max_steps):
        print(f"\nEpoch {step+1}/{max_steps}:")

        # Candidatos
        candidate_states = []

        for a in actions:
            x = np.hstack([S_t, a/90.0]).astype(np.float32)[None,:]
            S_pred = world_model.predict(x, verbose=0)[0]
            candidate_states.append({
                'action': a,
                'pred': S_pred,
                'novelty': novelty(S_pred, np.vstack(memory), n)
            })

        # Ordernar por novedad
        candidate_states = sorted(candidate_states, key=lambda x: x['novelty'], reverse=True)

        # for state in candidate_states:
        #     print(f"Accion: {state['action']}, Novedad: {state['novelty']:.4f}")

        best_action = candidate_states[0]['action']
        best_pred = candidate_states[0]['pred']

        # Comprobamos si la prediccion es meta (distancia al cilindro rojo por debajo de un umbral)
        if best_pred[1] < goal_thresh:
            print(f"Meta predicha con acción {best_action}")
            memory.append(best_pred.copy())
            break

        # Ejecutamos acción y comprobamos si lleva a un obstáculo
        for i, state in enumerate(candidate_states):
            print(f'Trying the candidate {i+1}')
            S_t1 = perform_main_action(robot, sim, state['action'], duration=0.5)
            if not undo_if_needed(robot, state['action']):
                memory.append(S_t1)
                S_t = S_t1
                break

        # Comprobar si el estado actual es meta
        if S_t[1] < goal_thresh:
            print(f"Meta real alcanzada en paso {step}")
            break

    return memory
