import pandas as pd
import numpy as np

HARTREE = 627.50947406

def calc_loss(df, type):
    ret = 0.0
    N = 0
    if (type != 'RMS' and type != 'MAE'):
        raise ValueError(type, 'not specified (inputs are RMS/MAE)')

    for seq in df:
        for atom in seq:
            N += 1
            targets = atom['targets']['Etot']
            predict = atom['predictions']['Etot'].numpy()
            diff = np.subtract(targets, predict)
            len = np.linalg.norm(diff)
            if (type == 'RMS'):
                len = len ** 2
            ret += len
    ret *= HARTREE      # Multiply here?
    if (type == 'RMS'):
        return (ret/N) ** 0.5
    return ret/N

# For testing
dataset_path = '../predicted_validation.pkl'
df = pd.read_pickle(dataset_path)

print(calc_loss(df, 'RMS'))
print(calc_loss(df, 'MAE'))