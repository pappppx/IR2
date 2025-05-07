import os
import pandas as pd

def merge_csv_files(input_dir: str, output_file: str, start: int = 1, end: int = 100):
    """
    Concatena CSVs numerados de start a end (inclusive) en input_dir
    y escribe el resultado en output_file.
    """
    dfs = []
    for i in range(start, end + 1):
        path = os.path.join(input_dir, f"{i}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
        else:
            print(f"Warning: '{i}.csv' no encontrado, se omite.")

    if not dfs:
        print("No hay archivos para concatenar.")
        return

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output_file, index=False)
    print(f"Se han unido {len(dfs)} archivos en '{output_file}'.")

if __name__ == "__main__":
    # Ajusta input_dir si tus CSVs están en otra carpeta
    merge_csv_files(input_dir=".", output_file="merged.csv", start=1, end=100)
