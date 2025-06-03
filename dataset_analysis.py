import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("datasets/world_dataset.csv")

    print(f"Dataset length: {len(df)}")

    # Extract position columns for each color (both t and t1)
    red_positions = pd.concat([df['red_position_t'], df['red_position_t1']])
    green_positions = pd.concat([df['green_position_t'], df['green_position_t1']])
    blue_positions = pd.concat([df['blue_position_t'], df['blue_position_t1']])

    # Compute statistics for each color
    stats = {}
    for color, series in [('Red', red_positions), ('Green', green_positions), ('Blue', blue_positions)]:
        stats[color] = {
            'min': series.min(),
            'max': series.max(),
            'mean': series.mean(),
            'std': series.std()
        }

    # Display the statistics
    for color, s in stats.items():
        print(f"{color} Distances:")
        print(f"  Min:  {s['min']:.6f}")
        print(f"  Max:  {s['max']:.6f}")
        print(f"  Mean: {s['mean']:.6f}")
        print(f"  Std:  {s['std']:.6f}\n")

    # Plot histograms separately for each color, arranged horizontally
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].hist(red_positions, bins=10)
    axes[0].set_title("Red Distances (t & t1)")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(green_positions, bins=10)
    axes[1].set_title("Green Distances (t & t1)")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Frequency")

    axes[2].hist(blue_positions, bins=10)
    axes[2].set_title("Blue Distances (t & t1)")
    axes[2].set_xlabel("Distance")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Compute absolute differences |t1 - t| for position columns
    red_abs_diff = (df['red_position_t1'] - df['red_position_t']).abs()
    green_abs_diff = (df['green_position_t1'] - df['green_position_t']).abs()
    blue_abs_diff = (df['blue_position_t1'] - df['blue_position_t']).abs()

    # Compute statistics for each color's absolute differences
    abs_diff_stats = {}
    for color, abs_diff_series in [('Red', red_abs_diff), ('Green', green_abs_diff), ('Blue', blue_abs_diff)]:
        abs_diff_stats[color] = {
            'min': abs_diff_series.min(),
            'max': abs_diff_series.max(),
            'mean': abs_diff_series.mean(),
            'std': abs_diff_series.std()
        }

    # Display the statistics
    for color, s in abs_diff_stats.items():
        print(f"{color} Absolute Position Difference |t1 - t|:")
        print(f"  Min:  {s['min']:.6f}")
        print(f"  Max:  {s['max']:.6f}")
        print(f"  Mean: {s['mean']:.6f}")
        print(f"  Std:  {s['std']:.6f}\n")

    # Plot histograms for each color's absolute differences, arranged horizontally
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].hist(red_abs_diff, bins=10)
    axes[0].set_title("Red |Δ Position|")
    axes[0].set_xlabel("Absolute Difference")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(green_abs_diff, bins=10)
    axes[1].set_title("Green |Δ Position|")
    axes[1].set_xlabel("Absolute Difference")
    axes[1].set_ylabel("Frequency")

    axes[2].hist(blue_abs_diff, bins=10)
    axes[2].set_title("Blue |Δ Position|")
    axes[2].set_xlabel("Absolute Difference")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
