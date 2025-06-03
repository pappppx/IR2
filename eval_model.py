import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from utils.world import _load_and_split


def main():

    utility_model = load_model("models/world/world_model.keras")
    utility_model.summary()

    X_train, X_test, y_train, y_test = _load_and_split("datasets/world_dataset.csv")

    y_pred = utility_model.predict(X_test, verbose=0)

    n=20

    true_red_pos = y_test[:n, 1]
    pred_red_pos = y_pred[:n, 1]

    steps = list(range(1, n + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(steps, true_red_pos, marker='o', label='Observed red distance')
    plt.plot(steps, pred_red_pos, marker='x', label='Predicted red distance')
    plt.xlabel(f'Step (first {n} test samples)')
    plt.ylabel('Red Distance')
    plt.title(f'Observed vs. Predicted Red Distance for First {n} Samples')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()