import pickle
from train_utility_model import train_utility_model

def main():
    # 1) Carga de las trazas previamente guardadas
    with open("all_traces_130.pkl", "rb") as f:
        all_traces = pickle.load(f)
    print(f"Cargadas {len(all_traces)} trazas para entrenamiento.")

    # 2) Entrenamiento del Utility Model
    model = train_utility_model(
        traces=all_traces,
        window=10,
        save_path="utility_model.keras"
    )

    print("Entrenamiento completado y modelo guardado.")

if __name__ == "__main__":
    main()
