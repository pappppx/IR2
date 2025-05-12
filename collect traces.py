from tensorflow.keras.models import load_model
from utility_utils import intrinsic_exploration_loop
from perceptions import get_simple_perceptions
from actions import perform_random_action
from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo
import pickle

# Carga tu modelo del mundo
world_model = load_model("models/1530-130.keras")

sim = RoboboSim('localhost'); sim.connect(); sim.wait(0.5)
rob = Robobo('localhost'); rob.connect(); rob.wait(0.5)
    
# Parámetros
actions   = [-90, -45, 0, 45, 90]
all_traces = []
episodes  = 20

for ep in range(episodes):
    print(f"\n=== Episodio {ep+1} ===")
    perform_random_action(rob)
    trace = intrinsic_exploration_loop(
        rob,
        sim,
        world_model,
        actions=actions,
        n=4.0,
        max_steps=100,
        goal_thresh=350.0
    )
    all_traces.append(trace)

    sim.resetSimulation()
    sim.wait(0.5)

print(all_traces)

# Guardar trazas
with open("all_traces_130.pkl", "wb") as f:
    pickle.dump(all_traces, f)

sim.disconnect()
rob.disconnect()

