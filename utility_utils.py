import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

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

def train_utility_model(traces, window=10, save_path="utility_model.keras"):
    X, y = prepare_utility_dataset(traces, window)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(y_train)

    model = Sequential([
        Input(shape=(6,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1,  activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
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
    diffs = memory[-30:] - candidate[np.newaxis, :]
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

import numpy as np
from perceptions import get_simple_perceptions
from actions     import perform_main_action

def intrinsic_exploration_loop2(robot, sim, world_model, actions,
                               n: float = 1.0,
                               max_steps: int = 100,
                               goal_thresh: float = 250.0):
    """
    Exploration loop que:
     1) Si el modelo del mundo predice meta alcanzada (red_dist < goal_thresh)
        para alguna acción, la ejecuta inmediatamente.
     2) En otro caso, elige la acción más novedosa (hasta probar 5),
        llama a perform_main_action; si da None (go_back), prueba la siguiente.
    """
    # 1) Estado inicial
    P0 = get_simple_perceptions(sim)
    S_t = np.array([
        P0['red_rotation'],  P0['red_position'],
        P0['green_rotation'],P0['green_position'],
        P0['blue_rotation'], P0['blue_position']
    ], dtype=np.float32)
    memory = [S_t.copy()]

    for step in range(max_steps):
        # 2) Generar todas las predicciones
        preds = []
        for a in actions:
            x = np.hstack([S_t, a/90.0]).astype(np.float32)[None,:]
            S_pred = world_model.predict(x)[0]
            preds.append((a, S_pred))

        # 3) ¿Alguna predicción alcanza la meta?
        goals = [(a, S_pred) for a, S_pred in preds if S_pred[1] < goal_thresh]
        if goals:
            # elegimos la predicción que deja la menor distancia al rojo
            best_action, best_pred = min(goals, key=lambda t: t[1][1])
            print(f"Meta predicha con acción {best_action}")
            # registramos el estado PREDICHO (opcional: o el real tras ejecutarlo)
            memory.append(best_pred.copy())
            # ejecutar la acción real y terminar
            S_t1 = perform_main_action(robot, sim, best_action, duration=0.5)
            if S_t1[1] < goal_thresh:
                print(f"Meta real alcanzada en paso {step}")
                break
        
        else:
            # 4) Si no hay meta, calculamos novedad para cada candidato
            novs = [(novelty(S_pred, np.vstack(memory), n), a, S_pred)
                    for a, S_pred in preds]
            # orden descendente por novedad
            novs.sort(key=lambda t: t[0], reverse=True)

            # 5) Probar hasta 5 acciones en orden de novedad
            S_t1 = None
            for _, act, _ in novs[:5]:
                S_try = perform_main_action(robot, sim, act, duration=0.5)
                sim.wait(0.1); robot.wait(0.1)
                if S_try is not None:
                    S_t1 = S_try
                    chosen = act
                    break

            if S_t1 is None:
                print(f"No hay acción válida en paso {step}, abortando.")
                break

            # 6) Guardar nuevo estado y actualizar
            memory.append(S_t1.copy())
            S_t = S_t1

            # 7) Comprobar meta real
            if S_t[1] < goal_thresh:
                print(f"Meta real alcanzada en paso {step}")
                break

    return memory

def intrinsic_exploration_loop3(robot, sim, world_model, actions,
                                n: float = 1.0,
                                max_steps: int = 100,
                                goal_thresh: float = 250.0):
    """
    Igual que intrinsic_exploration_loop2, pero:
      - La memoria sólo almacena posiciones [red_pos, green_pos, blue_pos].
      - La función de novedad actúa sobre ℝ³ de posiciones.
      - El world_model recibe como input rotaciones + posiciones.
    """
    # 1) Estado inicial completo
    P0 = get_simple_perceptions(sim)
    S_t = np.array([
        P0['red_rotation'],  P0['red_position'],
        P0['green_rotation'],P0['green_position'],
        P0['blue_rotation'], P0['blue_position']
    ], dtype=np.float32)  # shape (6,)

    # Memoria: sólo posiciones (índices 1, 3, 5)
    memory = [S_t[[1, 3, 5]].copy()]  # lista de arrays de shape (3,)

    for step in range(max_steps):
        # 2) Predecir todos los candidatos
        preds = []
        for a in actions:
            x = np.hstack([S_t, a/90.0]).astype(np.float32)[None, :]  # (1, 6 + acción)
            S_pred = world_model.predict(x)[0]  # (6,)
            preds.append((a, S_pred))

        # 3) ¿Alguna predicción alcanza la meta?
        # Usamos S_pred[1] = red_position
        goals = [(a, S_pred) for a, S_pred in preds if S_pred[1] < goal_thresh]
        if goals:
            # Elegir la que deja menor distancia al rojo
            best_action, best_pred = min(goals, key=lambda t: t[1][1])
            print(f"Meta predicha con acción {best_action}")
            # Guardar la posición predicha
            memory.append(best_pred[[1, 3, 5]].copy())
            # Ejecutar y salir
            S_t1 = perform_main_action(robot, sim, best_action, duration=0.5)
            if S_t1[1] < goal_thresh:
                print(f"Meta real alcanzada en paso {step}")
            break

        # 4) Calcular novedad en ℝ³
        M = np.vstack(memory)  # (m, 3)
        novs = []
        for a, S_pred in preds:
            pos_pred = S_pred[[1, 3, 5]]
            dists = np.linalg.norm(M - pos_pred[None, :], axis=1)
            nov_score = np.mean(dists ** n)
            novs.append((nov_score, a, S_pred))
        novs.sort(key=lambda t: t[0], reverse=True)

        # 5) Intentar las top-5 más novedosas
        S_t1 = None
        for _, act, _ in novs[:5]:
            S_full = perform_main_action(robot, sim, act, duration=0.5)
            sim.wait(0.1); robot.wait(0.1)
            if S_full is not None:
                S_t1 = S_full
                break

        if S_t1 is None:
            print(f"No hay acción válida en paso {step}, abortando.")
            break

        # 6) Actualizar estado y memoria
        S_t = S_t1
        memory.append(S_t[[1, 3, 5]].copy())

        # 7) Comprobar meta real
        if S_t[1] < goal_thresh:
            print(f"Meta real alcanzada en paso {step}")
            break

    return memory

def intrinsic_exploration_loop4(robot, sim, world_model, actions,
                                n: float = 1.0,
                                max_steps: int = 100,
                                goal_thresh: float = 250.0):
    """
    Igual a loop2, pero al probar cada top-5 acción:
      - Si evaded=True, añade S_main a memory y sigue.
      - Si evaded=False, acepta S_main, sale del bucle y continúa.
    """
    from perceptions import get_simple_perceptions

    # 1) Estado inicial
    P0 = get_simple_perceptions(sim)
    S_t = np.array([
        P0['red_rotation'],  P0['red_position'],
        P0['green_rotation'],P0['green_position'],
        P0['blue_rotation'], P0['blue_position']
    ], dtype=np.float32)
    memory = [S_t.copy()]

    for step in range(max_steps):
        # 2) Predicciones
        preds = []
        for a in actions:
            x = np.hstack([S_t, a/90.0]).astype(np.float32)[None,:]
            S_pred = world_model.predict(x)[0]
            preds.append((a, S_pred))

        # 3) Meta predicha?
        goals = [(a, S_pred) for a, S_pred in preds if S_pred[1] < goal_thresh]
        if goals:
            best_action, best_pred = min(goals, key=lambda t: t[1][1])
            print(f"Meta predicha con acción {best_action}")
            memory.append(best_pred.copy())
            S_main, ev = perform_main_action(robot, sim, best_action)
            if not ev and S_main[1] < goal_thresh:
                print(f"Meta real alcanzada en paso {step}")
            break

        # 4) Novedad
        novs = [(novelty(S_pred, np.vstack(memory), n), a, S_pred)
                for a, S_pred in preds]
        novs.sort(key=lambda t: t[0], reverse=True)

        # 5) Top-5 intentos
        S_t1 = None
        for score, act, _ in novs[:5]:
            S_main, ev = perform_main_action(robot, sim, act)
            sim.wait(0.1); robot.wait(0.1)

            if ev:
                # añadir la posición previa al retroceso
                memory.append(S_main.copy())
                continue

            # si no evadió, aceptamos esa transición
            S_t1 = S_main
            chosen = act
            break

        if S_t1 is None:
            print(f"No hay acción válida en paso {step}, dando marcha atrás.")
            robot.moveWheelsByTime(-20, -20, 1.0)
            continue

        # 6) Actualizar
        memory.append(S_t1.copy())
        S_t = S_t1

        # 7) Meta real
        if S_t[1] < goal_thresh:
            print(f"Meta real alcanzada en paso {step}")
            break

    return memory


# ——— intrinsic_exploration_loop5 ———
def intrinsic_exploration_loop5(robot, sim, world_model, actions,
                                n: float = 1.0,
                                max_steps: int = 100,
                                goal_thresh: float = 250.0):
    """
    Igual a loop3, pero al probar cada top-5 acción:
      - Si evaded=True, añade la POSICIÓN previa al retroceso a memory y sigue.
      - Si evaded=False, acepta el nuevo estado y sale del bucle de intentos.
    """
    from perceptions import get_simple_perceptions

    # 1) Estado inicial
    P0 = get_simple_perceptions(sim)
    S_t = np.array([
        P0['red_rotation'],  P0['red_position'],
        P0['green_rotation'],P0['green_position'],
        P0['blue_rotation'], P0['blue_position']
    ], dtype=np.float32)
    # memoria solo de posiciones [red_pos, green_pos, blue_pos]
    memory = [S_t[[1,3,5]].copy()]
    log = []
    
    for step in range(max_steps):
        # 2) Predicciones
        preds = []
        for a in actions:
            x = np.hstack([S_t, a/90.0]).astype(np.float32)[None,:]
            S_pred = world_model.predict(x)[0]
            preds.append((a, S_pred))

        # 3) Meta predicha?
        goals = [(a, p) for a, p in preds if p[1] < goal_thresh]
        if goals:
            best_action, best_pred = min(goals, key=lambda t: t[1][1])
            print(f"Meta predicha con acción {best_action}")
            memory.append(best_pred[[1,3,5]].copy())
            S_main, ev, loc = perform_main_action(robot, sim, best_action)
            if not ev and S_main[1] < goal_thresh:
                print(f"Meta real alcanzada en paso {step}")
            break

        # 4) Novedad en ℝ³
        M = np.vstack(memory)
        novs = []
        for a, S_pred in preds:
            pos = S_pred[[1,3,5]]
            score = np.mean(np.linalg.norm(M - pos[None,:], axis=1)**n)
            novs.append((score, a, S_pred))
        novs.sort(key=lambda t: t[0], reverse=True)

        # 5) Top-5 con registro de retrocesos
        new_pos = None
        for score, act, _ in novs[:5]:
            S_main, ev, loc = perform_main_action(robot, sim, act)
            sim.wait(0.1); robot.wait(0.1)

            if ev:
                # Logueamos la posición previa al retroceso
                log.append({
                    "step":   step,
                    "x":      loc["x"],
                    "y":      loc["y"],
                    "evaded": True
                })
                pos_back = S_main[[1,3,5]].copy()
                memory.append(pos_back)
                continue

            # aceptamos esta nueva percepción
            S_t = S_main
            new_pos = S_t[[1,3,5]].copy()
            # Podemos también loguear las transiciones exitosas:
            log.append({
                "step":   step,
                "x":      loc["x"],
                "y":      loc["y"],
                "evaded": False
            })
            break

        if new_pos is None:
            print(f"No hay acción válida en paso {step}, dando marcha atrás.")
            robot.moveWheelsByTime(-20, -20, 1.0)
            continue
    
        # 6) Guardar y seguir
        memory.append(new_pos)

        # 7) Meta real
        if new_pos[0] < goal_thresh:
            print(f"Meta real alcanzada en paso {step}")
            break

    return memory, log

def intrinsic_exploration_loop6(robot, sim, world_model, actions,
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

def intrinsic_exploration_loop_posrot(robot, sim, world_model, actions,
                                      n: float = 1.0,
                                      max_steps: int = 100,
                                      goal_thresh: float = 250.0):
    """
    Exploration loop con:
      - Input al world_model: [red_rot, red_pos, green_pos, blue_pos, action_norm]
      - Modelo predict solo posiciones [red_pos1, green_pos1, blue_pos1]
      - Novelty sobre posiciones (R^3)
      - Memory guarda solo posiciones
      - predicción de meta usa red_pos1 < goal_thresh
    """
    # 1) Leer percepción inicial
    P0 = get_simple_perceptions(sim)
    # extraer rotación del rojo y posiciones de los tres cilindros
    state_features = np.array([
        P0['red_rotation'],
        P0['red_position'],
        P0['green_position'],
        P0['blue_position']
    ], dtype=np.float32)  # shape (4,)
    # memoria solo de posiciones (índices 1,2,3)
    memory = [state_features[1:].copy()]  # list of (3,)

    for step in range(max_steps):
        # 2) Predecir todos los candidatos
        preds = []
        for a in actions:
            x = np.hstack([state_features, a/90.0]).astype(np.float32)[None,:]  # (1,5)
            pos_pred = world_model.predict(x)[0]  # (3,)
            preds.append((a, pos_pred))

        # 3) Meta predicha?
        goals = [(a, p) for a,p in preds if p[0] < goal_thresh]
        if goals:
            # menor red_pos1
            best_action, best_pred = min(goals, key=lambda t: t[1][0])
            print(f"Meta predicha con acción {best_action}")
            memory.append(best_pred.copy())
            # ejecutar acción real y terminar
            S_t1 = perform_main_action(robot, sim, best_action, duration=0.5)
            if S_t1[1] < goal_thresh:
                print(f"Meta real alcanzada en paso {step}")
                break
        
        else:
            # 4) Calcular novedad en R^3
            M = np.vstack(memory)  # (m,3)
            novs = [(np.mean(np.linalg.norm(M - p[1][None,:], axis=1)**n),
                    p[0], p[1]) for p in preds]
            # 2b) desplazamiento previsto:
            # delta = float(np.linalg.norm(pos_pred - state_features[1:]))

            # # 2c) score ajustado:
            # novs = novss / (delta + max_steps)
            novs.sort(key=lambda t: t[0], reverse=True)

            # 5) Intentar top-5
            new_pos = None
            for _, act, _ in novs[:5]:
                S_full = perform_main_action(robot, sim, act, duration=0.5)
                sim.wait(0.1); robot.wait(0.1)
                if S_full is not None:
                    # S_full: [red_rot, red_pos, green_rot, green_pos, blue_rot, blue_pos]
                    # actualizar state_features con nueva percepción
                    state_features = np.array([
                        S_full[0],  # red_rotation
                        S_full[1],  # red_position
                        S_full[3],  # green_position
                        S_full[5]   # blue_position
                    ], dtype=np.float32)
                    new_pos = state_features[1:].copy()
                    chosen = act
                    break
            if new_pos is None:
                print(f"No hay acción válida en paso {step}, dando marcha atrás.")
                robot.moveWheelsByTime(-20, -20, 1.0)
                continue

            # 6) Guardar en memoria posiciones y continuar
            memory.append(new_pos)
            # 7) Comprobar meta real
            if new_pos[0] < goal_thresh:
                print(f"Meta real alcanzada en paso {step}")
                break

    return memory



