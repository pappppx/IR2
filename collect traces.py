# from tensorflow.keras.models import load_model
from keras.models import load_model
from utility_utils import intrinsic_exploration_loop, intrinsic_exploration_loop2, intrinsic_exploration_loop_posrot, intrinsic_exploration_loop3, intrinsic_exploration_loop4, intrinsic_exploration_loop5, intrinsic_exploration_loop6
from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo
from actions import perform_random_action
import pickle, random, csv

sim = RoboboSim('localhost'); sim.connect(); sim.wait(0.5)
rob = Robobo('localhost'); rob.connect(); rob.wait(0.5)
    
# Parámetros
MAX_STEPS = 400
ACTIONS = [-90, -45, 0, 45, 90]
MEMORY_SIZE = [-5, -15, -20, -25, -30, -35, -40, -45]
EPISODES = 20
SPIN_SPEED = 20
FORWARD_SPEED = 20
N = 3.0
GOAL_THRESH = 350.0
MODEL_PATH = "114"

model = load_model(f"models/{MODEL_PATH}.keras")
all_traces = []
all_logs   = []

for M in MEMORY_SIZE:
    print(f"\n=== Tamaño de memoria: {abs(M)} ===")
    for ep in range(EPISODES):
        print(f"\n=== Episodio {ep+1} ===")

        # Movimiento inicial aleatorio
        perform_random_action(rob, SPIN_SPEED, FORWARD_SPEED)

        trace, log = intrinsic_exploration_loop6(
            rob,
            sim,
            model,
            actions=ACTIONS,
            n=N,
            max_steps=MAX_STEPS,
            goal_thresh=GOAL_THRESH
        )
        if trace is None:
            print("No se ha detectado meta en el episodio")
        else:
            all_traces.append(trace)
        
        for row in log:
            row["episode"] = ep+1
        all_logs.extend(log)
        
        sim.resetSimulation()
        sim.wait(3)
        rob.moveTiltTo(110,20)
        rob.moveTiltTo(90,20)

    # Guardar trazas
    with open(f"traces/traces_model_{MODEL_PATH}_M_{str(abs(M))}", "wb") as f:
        pickle.dump(all_traces, f)
    print("Trazas guardadas")

    # Guardar posiciones
    with open(f"positions/log_M_{str(abs(M))}.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode","step","x","z","evaded"])
        writer.writeheader()
        writer.writerows(all_logs)

sim.disconnect()
rob.disconnect()

