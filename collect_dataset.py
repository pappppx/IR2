from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo
from dataset_utils import collect_dataset
from merge_csv import merge_csv_files
import os
    
def main():
    sim = RoboboSim('localhost'); sim.connect(); sim.wait(0.5)
    rob = Robobo('localhost'); rob.connect(); rob.wait(0.5)

    print("=== Recolectando dataset SIMPLE ===")
    rob.moveTiltTo(110,20)
    rob.moveTiltTo(90,20)
    collect_dataset(rob, sim, n_samples=333, export_name="1.csv", simple=True)
    
    sim.disconnect()
    rob.disconnect()
    
    merge_csv_files(input_dir="datasets/", output_file="datasets/dataset_merge.csv", start=1, end=1)

if __name__ == '__main__':
    main()
