import matplotlib.pyplot as plt
import numpy as np

CYLINDER_POSITIONS = {'red': {'x': 600.0, 'y': 10.0, 'z': -600.0}, 'blue': {'x': 600.0, 'y': 10.0, 'z': 600.0}, 'green': {'x': -600.0, 'y': 10.0, 'z': -600.0}}


def plot_training_history(history):

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.yscale('log')
    plt.title('Model Loss Over Epochs (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show() 


def plot_position_scatter(df,
                          episode=None,
                          figsize=(6, 6),
                          point_size=10,
                          alpha=0.6,
                          last_n=None,
                          plot_cylinder=False,
                          scatter_path=None):

    if episode is not None:
        df = df[df['episode'] == episode]

    if last_n is not None:
        if episode is not None:
            df = df.tail(last_n)
        else:
            df = df.groupby('episode').tail(last_n)

    plt.figure(figsize=figsize)

    if plot_cylinder:
        for color, pos in CYLINDER_POSITIONS.items():
            plt.scatter(
                pos['x'], pos['z'],
                s=point_size * 10,
                c=color,
                edgecolors='black',
                alpha=1
            )
    plt.scatter(
        df['x'], df['z'],
        s=point_size,
        alpha=alpha,
        c='blue',
        edgecolors='none'
    )

    title = "Distribución de posiciones"
    if episode is not None:
        title += f" (episodio {episode})"
    plt.title(title)
    plt.xlabel("X (mm)")
    plt.ylabel("Z (mm)")
    plt.grid(True)
    plt.tight_layout()

    if scatter_path:
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print(f"Scatter guardado en '{scatter_path}'")
    else:
        plt.show()


def plot_distance_vs_utility(epoch_logs):

    plt.figure(figsize=(8, 6))
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, single_log in enumerate(epoch_logs):
        distances = np.array([entry['distance'] for entry in single_log])
        utilities = np.array([entry['utility_score'] for entry in single_log])

        color = cmap[idx % len(cmap)]
        plt.scatter(distances, utilities, color=color, alpha=0.7, edgecolors='k', label=f"Ep. {idx+1}")

    plt.xlabel("Distance (sensor reading)")
    plt.ylabel("Utility score (predicción)")
    plt.title("Distance vs. Utility Score (coloreado por episodio)")
    plt.grid(True)
    plt.legend(title="Episodios")
    plt.tight_layout()
    plt.show()