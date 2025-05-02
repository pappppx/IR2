from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo
from dataset_utils import collect_dataset
    
def main():
    sim = RoboboSim('localhost'); sim.connect(); sim.wait(0.5)
    rob = Robobo('localhost'); rob.connect(); rob.wait(0.5)

    print("=== Recolectando dataset SIMPLE ===")
    collect_dataset(rob, sim, n_samples=100, export_name="None3.csv", simple=True)

    sim.disconnect()
    rob.disconnect()

if __name__ == '__main__':
    main()
