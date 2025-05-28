import pickle
from utils.utility import train_utility_model

def main():
    # Carga de las trazas previamente guardadas
    with open("traces/traces_M_20v2.pkl", "rb") as f:
        all_traces = pickle.load(f)
        
    print(f"Cargadas {len(all_traces)} trazas para entrenamiento.")

    # Entrenamiento del Utility Model
    train_utility_model(
        traces=all_traces,
        window=10,
        save_path="models/utility/utility_model7.keras"
    )

if __name__ == "__main__":
    main()
