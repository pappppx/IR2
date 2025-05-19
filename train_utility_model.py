import pickle
from utility_utils import train_utility_model

def main():
    # Carga de las trazas previamente guardadas
    with open("traces/traces_model_114_M_5.pkl", "rb") as f:
        all_traces = pickle.load(f)
        
    print(f"Cargadas {len(all_traces)} trazas para entrenamiento.")

    # Entrenamiento del Utility Model
    train_utility_model(
        traces=all_traces,
        window=20,
        save_path="models/utility/utility_model.keras"
    )

if __name__ == "__main__":
    main()
