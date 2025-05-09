from tensorflow.keras.models import load_model
from utility_utils import intrinsic_exploration_loop
from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo

# 1) Carga tu modelo del mundo
world_model = load_model("1530-130.keras")

sim = RoboboSim('localhost'); sim.connect(); sim.wait(0.5)
rob = Robobo('localhost'); rob.connect(); rob.wait(0.5)
    
# 2) Parámetros
actions   = [-90, -45, 0, 45, 90]
all_traces = []
episodes  = 15

for ep in range(episodes):
    print(f"\n=== Episodio {ep+1} ===")
    trace = intrinsic_exploration_loop(
        rob, sim, world_model,
        actions=actions,
        n=1.0,
        max_steps=100,
        goal_thresh=350.0
    )
    all_traces.append(trace)
    # reinicia simulador/robot aquí si hace falta

sim.disconnect()
rob.disconnect()