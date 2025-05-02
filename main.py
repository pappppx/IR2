from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo
from model_utils import train_simple_model, train_complex_model
from dataset_utils import collect_dataset

def reset_sim(sim):
    sim.resetSimulation()
    sim.wait(0.5)
    
def main():
    sim = RoboboSim('localhost'); sim.connect(); sim.wait(0.5)
    rob = Robobo('localhost'); rob.connect(); rob.wait(0.5)

    i = 1  # 1 = Simple, 2 = Complejo, 0 = Ambos

    if i in [1, 0]:
        print("=== Recolectando dataset SIMPLE ===")
        dataset_simple = collect_dataset(rob, sim, n_samples=1000, export_name="simple_dataset.csv", simple=True)
        reset_sim(sim)
        train_simple_model(dataset_simple)

    if i in [2, 0]:
        print("=== Recolectando dataset COMPLEJO ===")
        dataset_complex = collect_dataset(rob, sim, n_samples=3 if i == 2 else 5, export_name="complex_dataset.csv", simple=False)
        reset_sim(sim)
        train_complex_model(dataset_complex)

    sim.disconnect()
    rob.disconnect()

if __name__ == '__main__':
    main()