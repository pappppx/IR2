from keras.models import load_model
from utils.utility import intrinsic_exploration_loop
from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo
from utils.actions import perform_random_action
import pickle, random, csv

# Directorios
WORLD_MODEL_PATH = "models/world/world_model.keras"
POSITIONS_PATH = "positions/"
TRACES_PATH = "traces/"

# Parámetros
MAX_STEPS = 400
ACTIONS = [-90, -45, 0, 45, 90]
MEMORY_SIZE = [15]
EPISODES = 40
SPIN_SPEED = 20
FORWARD_SPEED = 20
N = 3.0
GOAL_THRESH = 250.0

def main():
    sim = RoboboSim('localhost'); sim.connect()
    rob = Robobo('localhost'); rob.connect()
    
    model = load_model(f"{WORLD_MODEL_PATH}")
    all_traces = []
    all_logs   = []

    for M in MEMORY_SIZE:
        print(f"\n=== Tamaño de memoria: {abs(M)} ===")
        ep = 0
        while ep < EPISODES:
            print(f"\n=== Episodio {ep+1} ===")

            perform_random_action(rob, SPIN_SPEED, FORWARD_SPEED)

            trace, log = intrinsic_exploration_loop(
                rob,
                sim,
                model,
                actions=ACTIONS,
                m=M,
                n=N,
                max_steps=MAX_STEPS,
                goal_thresh=GOAL_THRESH
            )

            if trace is not None:

                ep += 1
                all_traces.append(trace)
    
                for row in log:
                    row["episode"] = ep
                all_logs.extend(log)
            
            sim.resetSimulation()
            sim.wait(1)

        with open(f"{TRACES_PATH}traces_M_{str(M)}v4.pkl", "wb") as f:
            pickle.dump(all_traces, f)

        with open(f"{POSITIONS_PATH}log_M_{str(M)}v4.csv","w",newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode","step","x","z","evaded"])
            writer.writeheader()
            writer.writerows(all_logs)

    sim.disconnect()
    rob.disconnect()

if __name__ == "__main__":
    main()