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
    
def transform_dataset(input_csv: str, output_csv: str):
    """
    Lee el CSV original con columnas:
      red_rotation_t, red_position_t,
      green_rotation_t, green_position_t,
      blue_rotation_t, blue_position_t,
      action,
      red_rotation_t1, red_position_t1,
      green_rotation_t1, green_position_t1,
      blue_rotation_t1, blue_position_t1

    Escribe un CSV nuevo donde por cada fila original
    genera 3 filas (una por cilindro), con columnas:
      angle_t, dist_t, action, angle_t1, dist_t1
    """
    df = pd.read_csv(input_csv)

    records = []
    for idx, row in df.iterrows():
        for color in ('red', 'green', 'blue'):
            records.append({
                'angle_t':  row[f'{color}_rotation_t'],
                'dist_t':   row[f'{color}_position_t'],
                'action':   row['action'],
                'angle_t1': row[f'{color}_rotation_t1'],
                'dist_t1':  row[f'{color}_position_t1'],
            })

    df2 = pd.DataFrame.from_records(records,
                                   columns=['angle_t','dist_t','action','angle_t1','dist_t1'])
    df2.to_csv(output_csv, index=False)
    print(f"Transformado: {len(df2)} filas escritas en '{output_csv}'")
