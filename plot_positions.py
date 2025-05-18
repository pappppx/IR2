import pandas as pd
import matplotlib.pyplot as plt

def plot_position_scatter(csv_path,
                          episode=None,
                          figsize=(6,6),
                          point_size=10,
                          alpha=0.6,
                          scatter_path=None):
    """
    Dibuja un scatter plot de las posiciones (x,y) registradas en el CSV.

    Parámetros:
    -----------
    csv_path : str
        Ruta al CSV con columnas 'episode','x','y','evaded'.
    episode : int o None
        Si es int, filtra solo ese episodio. Si es None, dibuja todos.
    figsize : tuple
        Tamaño de la figura.
    point_size : int
        Tamaño de cada punto.
    alpha : float
        Transparencia de los puntos, en [0,1].
    scatter_path : str o None
        Si se especifica, ruta donde guardar el PNG; si no, muestra inline.
    """
    # 1) Carga y filtra
    df = pd.read_csv(csv_path)
    if episode is not None:
        df = df[df['episode'] == episode]

    # 2) Scatter
    plt.figure(figsize=figsize)
    plt.scatter(df['x'], df['z'],
                s=point_size,
                alpha=alpha,
                c='blue', edgecolors='none')

    # 3) Etiquetas y título
    title = "Distribución de posiciones"
    if episode is not None:
        title += f" (episodio {episode})"
    plt.title(title)
    plt.xlabel("X (mm)")
    plt.ylabel("Z (mm)")
    plt.grid(True)
    plt.tight_layout()

    # 4) Guardar o mostrar
    if scatter_path:
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print(f"Scatter guardado en '{scatter_path}'")
    else:
        plt.show()

plot_position_scatter('positions/log_M-5.csv')

# Scatter solo del episodio 5
# plot_position_scatter('datasets/positions_log.csv', episode=5)