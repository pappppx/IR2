import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from utils.perceptions import get_perception_vector
from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo
from utils.actions import perform_random_action, perform_action
from utils.utility import test_model
from utils.visualization import plot_distance_vs_utility

# Directorios
MODEL_PATH = "models/"
POSITIONS_PATH = "positions/"
TRACES_PATH = "traces/"

# Parámetros
MAX_STEPS = 100
ACTIONS = [-90, -45, 0, 45, 90]
SPIN_SPEED = 20
FORWARD_SPEED = 20
GOAL_THRESH = 250.0
EPISODES = 10


def main():
    sim = RoboboSim('localhost'); sim.connect()
    rob = Robobo('localhost'); rob.connect()

    utility_model = load_model("models/utility/utility_model9.keras")
    world_model = load_model("models/world/world_model.keras")

    n_moves = []
    epoch_logs = []

    for i in range(EPISODES):
        print(f"\nExecution {i+1}:")
        perform_random_action(rob, SPIN_SPEED, FORWARD_SPEED)

        moves, log = test_model(
            robot=rob,
            sim=sim,
            world_model=world_model,
            utility_model=utility_model,
            actions=ACTIONS,
            max_steps=MAX_STEPS,
            goal_thresh=GOAL_THRESH
        )
        n_moves.append(moves)
        epoch_logs.append(log)

        sim.resetSimulation()
        sim.wait(1)

    print(f"\nMedia de movimientos: {np.mean(n_moves)}")
    plot_distance_vs_utility(epoch_logs)

if __name__ == "__main__":
    main()