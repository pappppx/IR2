import pickle
from utils.utility import train_utility_model

def main():

    with open("traces/traces_M_15v3.pkl", "rb") as f:
        all_traces = pickle.load(f)

    train_utility_model(
        traces=all_traces,
        epochs=300,
        window=20,
        save_path="models/utility/utility_model10.keras"
    )

if __name__ == "__main__":
    main()
