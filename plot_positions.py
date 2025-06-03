from utils.visualization import plot_position_scatter
import pandas as pd

def main():

    df = pd.read_csv('positions/log_M_15v3.csv')  
    plot_position_scatter(df, plot_cylinder=True, last_n=15)

if __name__ == "__main__":
    main()
