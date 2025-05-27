import time
from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo
from utils.perceptions import get_simple_perceptions, angle_from_sin_cos


def main():
    sim = RoboboSim('localhost'); sim.connect(); sim.wait(0.5)
    rob = Robobo('localhost'); rob.connect(); rob.wait(0.5)

    rob.moveWheels(2, -2)
    while True:
        P = get_simple_perceptions(sim)
        print(f"angle: {angle_from_sin_cos(P['red_sin'], P['red_cos'])}, sin {P['red_sin']}, cos {P['red_cos']} distance: {P['red_position']}")
        time.sleep(0.5)

if __name__ == "__main__":
    main()