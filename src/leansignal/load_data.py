import pandas as pd
from scipy.io import arff
from pathlib import Path

PATH_ROOT = Path(__file__).resolve().parent.parent.parent

def load_data():
    data, _ = arff.loadarff(f'{PATH_ROOT}/data/raw/3year.arff')
    df = pd.DataFrame(data)
    return df