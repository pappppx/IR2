from robobopy.utils.IR import IR
from utils.actions import perform_simple_action, perform_continuous_action, sample_random_continuous_action, avoid_if_needed
from utils.perceptions import get_simple_perceptions
import random
import os
import pandas as pd


DEFAULT_CSV_PATH = "datasets/"

def collect_simple_dataset(robot, sim, n_samples):
    dataset = []

    for i in range(n_samples):

        P_t = get_simple_perceptions(sim)
        robot.wait(0.1)
        action = random.choice([-90, -45, 0, 45, 90])
        print(f"Epoch {i}, action {action}")
        angle = perform_simple_action(robot, action, duration=0.5)
        robot.wait(0.25)
        sim.wait(0.25)
        
        if angle == None:
            robot.wait(0.1)
            
        else:
            P_t1 = get_simple_perceptions(sim)
            if avoid_if_needed(robot):
                robot.wait(0.1)
            
            else:
                dataset.append({
            # red cylinder features
            "red_sin_t":   P_t["red_sin"],
            "red_cos_t":   P_t["red_cos"],
            "red_dist_t":  P_t["red_dist"],

            # green cylinder features
            "green_sin_t":  P_t["green_sin"],
            "green_cos_t":  P_t["green_cos"],
            "green_dist_t": P_t["green_dist"],

            # blue cylinder features
            "blue_sin_t":   P_t["blue_sin"],
            "blue_cos_t":   P_t["blue_cos"],
            "blue_dist_t":  P_t["blue_dist"],

            # the action taken
            "action": action,

            # red cylinder after
            "red_sin_t1":   P_t1["red_sin"],
            "red_cos_t1":   P_t1["red_cos"],
            "red_dist_t1":  P_t1["red_dist"],

            # green cylinder after
            "green_sin_t1":  P_t1["green_sin"],
            "green_cos_t1":  P_t1["green_cos"],
            "green_dist_t1": P_t1["green_dist"],

            # blue cylinder after
            "blue_sin_t1":   P_t1["blue_sin"],
            "blue_cos_t1":   P_t1["blue_cos"],
            "blue_dist_t1":  P_t1["blue_dist"],
        })

    return pd.DataFrame(dataset)


def collect_dataset(robot, sim, n_samples=50, export_name=None, simple=True):

    if export_name is not None:
        try:
            return pd.read_csv(DEFAULT_CSV_PATH + export_name)
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"Warning: no pude cargar CSV, lo regenero ({e})")

    dataset = collect_simple_dataset(robot, sim, n_samples)

    if not os.path.exists(DEFAULT_CSV_PATH):
        os.makedirs(DEFAULT_CSV_PATH)
    dataset.to_csv(DEFAULT_CSV_PATH + export_name, index=False)
    
    return dataset
