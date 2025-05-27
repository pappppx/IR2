from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo
from utils.dataset import collect_dataset
from merge_csv import merge_csv_files
import os
    
def main():
    sim = RoboboSim('localhost'); sim.connect(); sim.wait(0.5)
    rob = Robobo('localhost'); rob.connect(); rob.wait(0.5)

    TOTAL_SAMPLES = 800
    BATCH_SIZE = 400

    print("=== Recolectando dataset SIMPLE ===")
    for i in range(TOTAL_SAMPLES // BATCH_SIZE):

        rob.moveTiltTo(110,20)
        rob.moveTiltTo(90,20)
        collect_dataset(rob, sim, n_samples=400, export_name=f"{i}.csv", simple=True)
    
    sim.disconnect()
    rob.disconnect()
    
    merge_csv_files(input_dir="datasets/", output_file="datasets/sin_cos_dataset.csv", start=1, end=2)

if __name__ == '__main__':
    main()
