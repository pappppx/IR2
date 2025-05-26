from utils.visualization import plot_position_scatter

def main():
    plot_position_scatter('positions/log_M_20v2.csv')

    # Scatter solo del episodio 5
    # plot_position_scatter('datasets/positions_log.csv', episode=5)

if __name__ == "__main__":
    main()
