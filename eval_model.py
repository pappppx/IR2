import pickle
from keras.models import load_model
from utils.utility import prepare_utility_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    with open("traces/traces_model_114_M_30.pkl", "rb") as f:
        all_traces = pickle.load(f)

    utility_model = load_model("models/utility/utility_model.keras")

    X, y = prepare_utility_dataset(all_traces, 10)
    y_pred = utility_model.predict(X).flatten()
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"Utility model MAE: {mae:.4f}")
    print(f"Utility model MSE: {mse:.4f}")


if __name__ == "__main__":
    main()