from robobopy.utils.IR import IR
from actions import perform_simple_action, perform_continuous_action, sample_random_continuous_action, avoid_if_needed
from perceptions import get_simple_perceptions, scan_for_cylinders
import random
import os
import pandas as pd


DEFAULT_CSV_PATH = "datasets/"

def collect_simple_dataset(robot, sim, n_samples):
    dataset = []

    for i in range(n_samples):

        print(f"Epoch {i}")

        # 1) Medimos P(t) en modo simple
        P_t = get_simple_perceptions(sim)
        
        # 2) Escogemos y ejecutamos acción discreta
        accion = random.choice([-90, -45, 0, 45, 90])
        perform_simple_action(robot, accion, duration=2.0)
        
        # 3) Medimos P(t+1)
        P_t1 = get_simple_perceptions(sim)
        
        avoid_if_needed(robot)
        
        # 4) Guardamos la muestra
        dataset.append({
            "red_rotation_t":   P_t["red_rotation"],
            "red_position_t":   P_t["red_position"],
            "green_rotation_t": P_t["green_rotation"],
            "green_position_t": P_t["green_position"],
            "blue_rotation_t":  P_t["blue_rotation"],
            "blue_position_t":  P_t["blue_position"],

            "action":  accion,

            "red_rotation_t1":   P_t1["red_rotation"],
            "red_position_t1":   P_t1["red_position"],
            "green_rotation_t1": P_t1["green_rotation"],
            "green_position_t1": P_t1["green_position"],
            "blue_rotation_t1":  P_t1["blue_rotation"],
            "blue_position_t1":  P_t1["blue_position"],
        })

    return pd.DataFrame(dataset)

def collect_complex_dataset(robot, sim, n_samples):
    dataset = []

    for i in range(n_samples):
        # 1) Medimos P(t) en modo complejo
        P_t = scan_for_cylinders(robot, wheel_speed=6)
        
        # 2) Escogemos y ejecutamos acción continua
        left, right = sample_random_continuous_action(max_power=20)
        perform_continuous_action(robot, left, right, duration=2.0)
        sim.wait(0.1)
        robot.wait(0.1)
        
        # 3) Medimos P(t+1) de nuevo en modo complejo
        P_t1 = scan_for_cylinders(robot, wheel_speed=6)
        
        avoid_if_needed(robot)
        
        # 4) Guardamos la muestra: acción es el par (left, right)
        row = {}

        for color in ["red","green","blue"]:
            row[f"{color}_x_t"]    = P_t[color]["x"]
            row[f"{color}_y_t"]    = P_t[color]["y"]
            row[f"{color}_size_t"] = P_t[color]["size"]

        row["left_t"], row["right_t"] = (left, right)

        for color in ["red","green","blue"]:
            row[f"{color}_x_t1"]    = P_t1[color]["x"]
            row[f"{color}_y_t1"]    = P_t1[color]["y"]
            row[f"{color}_size_t1"] = P_t1[color]["size"]

        dataset.append(row)

    return pd.DataFrame(dataset)

def collect_dataset(robot, sim, n_samples=50, export_name=None, simple=True):
    """
    Recolecta n muestras de la forma (P_t, acción, P_t1).
    
    - simple=True: usa el espacio discreto de ángulos y mide P_simple.
    - simple=False: usa potencias continuas y mide P_complex.
    """

    if export_name is not None:
        try:
            return pd.read_csv(DEFAULT_CSV_PATH + export_name)
        except FileNotFoundError as e:
            print(f"Error: {e}")

    if simple:
        dataset = collect_simple_dataset(robot, sim, n_samples)
    else:
        dataset = collect_complex_dataset(robot, sim, n_samples)

    if not os.path.exists(DEFAULT_CSV_PATH):
        os.makedirs(DEFAULT_CSV_PATH)
    dataset.to_csv(DEFAULT_CSV_PATH + export_name, index=False)
    
    return dataset