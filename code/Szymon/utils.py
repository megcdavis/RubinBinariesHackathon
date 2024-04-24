import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy.table import Table
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def make_clf_report(y_true, y_pred, df):
    y_class = np.argmax(y_pred, axis=1)
    print(classification_report(y_true, y_class))
    plot_acc(y_true, y_class, df['AMP'])


def plot_acc(y_true, y_pred, amplitude):
    bins = np.concatenate([np.arange(min(amplitude), max(amplitude), 0.1), [max(amplitude)]])
    n_obs, acc_scores = [], []
    for i in range(len(bins) - 1):
        idx = (bins[i] < amplitude) & (amplitude < bins[i+1])
        n_obs.append(sum(idx))
        acc_scores.append(accuracy_score(y_true[idx], y_pred[idx]))
    
    bin_centers = bins[:-1] + np.diff(bins)
    plt.plot(bin_centers, acc_scores)
    # plt.plot(bin_centers, n_obs)


def get_train_data(n=None):
    # Read all the data
    file_name = '/home/sjnakoneczny/data/LSST/binaries/X.npy'
    X = np.load(file_name)

    file_name = '/home/sjnakoneczny/data/LSST/binaries/df.csv'
    df = pd.read_csv(file_name)

    if n:
        X = X[:n]
        df = df.head(n)

    y = df['AMP'] > 0.5

    # Take only the first 3000 values
    X = X[:, :3000]

    # Standardize
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    for i in range(len(X)):
        X[i] = (X[i] - mean[i]) / std[i]

    # Make train test split
    return train_test_split(X, y, test_size=0.1, random_state=3462)


def read_fits_files(n_files=None):
    Xs, dfs = [], []
    files = glob.glob('/home/sjnakoneczny/data/LSST/binaries/fits/file_*.fits')
    if n_files:
        files = files[:n_files]

    for file in tqdm(files):
        table = Table.read(file, format='fits')
        Xs.append(table.to_pandas().drop('time', axis=1).to_numpy().transpose())
        dfs.append(pd.DataFrame(table.meta))
        
    return np.concatenate(Xs), pd.concat(dfs)
