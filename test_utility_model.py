import numpy as np
from keras.models import load_model
from utils.perceptions import get_perception_vector
from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo
from utils.actions import perform_main_action, perform_random_action

# Directorios
MODEL_PATH = "models/"
POSITIONS_PATH = "positions/"
TRACES_PATH = "traces/"

# Parámetros
MAX_STEPS = 100
ACTIONS = [-90, -45, 0, 45, 90]
SPIN_SPEED = 20
FORWARD_SPEED = 20
GOAL_THRESH = 350.0

def test_model(
    robot, 
    sim, 
    world_model,
    utility_model,
    actions,
    max_steps: int = 100,
    goal_thresh: float = 250.0):

    for step in range(max_steps):

        s_t0 = get_perception_vector(sim)

        world_model_predictions = []
        for a in actions:
            x = np.hstack([s_t0, a/90.0]).astype(np.float32)[None,:]
            S_pred = world_model.predict(x, verbose=0)[0]
            world_model_predictions.append((a, S_pred))

        utility_model_predictions = []
        for a, S_pred in world_model_predictions:
            utility_score = utility_model.predict(S_pred[None,:], verbose=0)[0]
            utility_model_predictions.append({
                'score': utility_score[0],
                'action': a,
                'S_pred': S_pred
            })

        utility_model_predictions.sort(key=lambda x: x["score"], reverse=False)

        for element in utility_model_predictions:
            
            _, evade, _ = perform_main_action(robot, sim, element['action'])
            sim.wait(0.1); robot.wait(0.1)

            if evade: continue
            break

        s_t1 = get_perception_vector(sim)

        if s_t1[1] < goal_thresh:
            print(f"Meta real alcanzada en paso {step}")
            return step
        
    print("No se ha alcanzado la meta")


def main():
    sim = RoboboSim('localhost'); sim.connect(); sim.wait(0.5)
    rob = Robobo('localhost'); rob.connect(); rob.wait(0.5)

    utility_model = load_model("models/utility/utility_model.keras")
    world_model = load_model("models/world/world_model.keras")

    n_moves = []

    for i in range(10):
        perform_random_action(rob, SPIN_SPEED, FORWARD_SPEED)

        moves = test_model(
            robot=rob,
            sim=sim,
            world_model=world_model,
            utility_model=utility_model,
            actions=ACTIONS,
            max_steps=MAX_STEPS,
            goal_thresh=GOAL_THRESH
        )
        n_moves.append(moves)

        sim.resetSimulation()
        sim.wait(1)

    print(f"Media de movimientos: {np.mean(n_moves)}")

if __name__ == "__main__":
    main()