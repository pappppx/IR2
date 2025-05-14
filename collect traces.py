from tensorflow.keras.models import load_model
from utility_utils import intrinsic_exploration_loop, intrinsic_exploration_loop2, intrinsic_exploration_loop_posrot
from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo
import pickle

# 1) Carga tu modelo del mundo
world_model = load_model("models/96.keras")

sim = RoboboSim('localhost'); sim.connect(); sim.wait(0.5)
rob = Robobo('localhost'); rob.connect(); rob.wait(0.5)
    
# 2) Parámetros
actions   = [-90, -45, 0, 45, 90]
all_traces = []
episodes  = 20

for ep in range(episodes):
    print(f"\n=== Episodio {ep+1} ===")
    trace = intrinsic_exploration_loop_posrot(
        rob, sim, world_model,
        actions=actions,
        n=3.0,
        max_steps=100,
        goal_thresh=350.0
    )
    all_traces.append(trace)
    # reinicia simulador/robot aquí si hace falta
    sim.resetSimulation()
    sim.wait(3)
    rob.moveTiltTo(110,20)
    rob.moveTiltTo(90,20)

# al final de collect_traces.py, tras llenar all_traces:
with open("models/all_traces_96_4.pkl", "wb") as f:
    pickle.dump(all_traces, f)
print("Trazas guardas.pkl")

world_model2 = load_model("models/114.keras")
all_traces = []
episodes  = 20

for ep in range(episodes):
    print(f"\n=== Episodio {ep+1} ===")
    trace = intrinsic_exploration_loop2(
        rob, sim, world_model2,
        actions=actions,
        n=15.0,
        max_steps=100,
        goal_thresh=350.0
    )
    all_traces.append(trace)
    # reinicia simulador/robot aquí si hace falta
    sim.resetSimulation()
    sim.wait(3)
    rob.moveTiltTo(110,20)
    rob.moveTiltTo(90,20)

# al final de collect_traces.py, tras llenar all_traces:
with open("models/all_traces_114_4.pkl", "wb") as f:
    pickle.dump(all_traces, f)
print("Trazas guardadas")

sim.disconnect()
rob.disconnect()

