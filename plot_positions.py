from utils.visualization import plot_position_scatter
import pandas as pd

def main():

    df = pd.read_csv('positions/log_M_30.csv')
    plot_position_scatter(df, plot_cylinder=True)

if __name__ == "__main__":
    main()
