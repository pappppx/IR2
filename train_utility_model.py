import pickle
from utils.utility import train_utility_model

def main():

    with open("traces/traces_M_15v3.pkl", "rb") as f:
        all_traces = pickle.load(f)
        
    print(f"Cargadas {len(all_traces)} trazas, con media de {sum(len(t) for t in all_traces)/len(all_traces):.2f} pasos")

    train_utility_model(
        traces=all_traces,
        epochs=300,
        window=15,
        save_path="models/utility/utility_model6.keras"
    )

if __name__ == "__main__":
    main()
