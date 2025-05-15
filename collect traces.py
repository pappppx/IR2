# from tensorflow.keras.models import load_model
from keras.models import load_model
from utility_utils import intrinsic_exploration_loop, intrinsic_exploration_loop2, intrinsic_exploration_loop_posrot, intrinsic_exploration_loop3, intrinsic_exploration_loop4, intrinsic_exploration_loop5
from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo
import pickle, random

sim = RoboboSim('localhost'); sim.connect(); sim.wait(0.5)
rob = Robobo('localhost'); rob.connect(); rob.wait(0.5)
    
# 2) Parámetros
actions   = [-90, -45, 0, 45, 90]
all_traces = []
episodes  = 20

# 1) Carga tu modelo del mundo
# world_model = load_model("models/96.keras")

# for ep in range(episodes):
#      print(f"\n=== Episodio {ep+1} ===")
#      trace = intrinsic_exploration_loop_posrot(
#          rob, sim, world_model,
#          actions=actions,
#          n=6.0,
#          max_steps=100,
#          goal_thresh=350.0
#      )
#      all_traces.append(trace)
#      # reinicia simulador/robot aquí si hace falta
#      sim.resetSimulation()
#      sim.wait(3)
#      rob.moveTiltTo(110,20)
#      rob.moveTiltTo(90,20)

# # al final de collect_traces.py, tras llenar all_traces:
# with open("models/all_traces_96_4.pkl", "wb") as f:
#     pickle.dump(all_traces, f)
# print("Trazas guardas.pkl")

world_model2 = load_model("models/114.keras")
all_traces = []
episodes  = 20

spin_speed    = 20
forward_speed = 20

for ep in range(episodes):
    print(f"\n=== Episodio {ep+1} ===")

    # --- Pre-movimiento aleatorio ---
    # Ángulo aleatorio en grados [-180, +180]
    rand_angle = random.uniform(0.0, 360.0)
    # Duración proporcional al ángulo (1.75 s equivale a 180°)
    t_turn = abs(rand_angle) / 180.0 * 1.75

    if rand_angle > 0:
        rob.moveWheelsByTime(-spin_speed, spin_speed, t_turn)
    elif rand_angle < 0:
        rob.moveWheelsByTime(spin_speed, -spin_speed, t_turn)
    # si rand_angle == 0, no gira
    rob.wait(0.1)

    # Avance aleatorio entre 0.5 y 1.5 segundos
    rand_duration = random.uniform(0.5, 1.5)
    rob.moveWheelsByTime(forward_speed, forward_speed, rand_duration)
    rob.wait(0.1)
    # — fin pre-movimiento —
    trace = intrinsic_exploration_loop4(
        rob, sim, world_model2,
        actions=actions,
        n=3.0,
        max_steps=150,
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

