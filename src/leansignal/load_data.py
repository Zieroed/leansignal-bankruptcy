import pandas as pd
from scipy.io import arff
from pathlib import Path

PATH_ROOT = Path(__file__).resolve().parent.parent.parent

def load_data():
    data, _ = arff.loadarff(f'{PATH_ROOT}/data/raw/3year.arff')
    df = pd.DataFrame(data)
    df = df.map(lambda x: int(x.decode()) if isinstance(x, bytes) else x)
    return df

def save_data(df, file_name):
    file_path = Path(f"{PATH_ROOT}/results/{file_name}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    return