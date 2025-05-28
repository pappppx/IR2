import pandas as pd
from utils.visualization import plot_position_scatter

def main():
    df = pd.read_csv('positions/log_M_15v4.csv')
    plot_position_scatter(df, plot_cylinder=True)


if __name__ == "__main__":
    main()
