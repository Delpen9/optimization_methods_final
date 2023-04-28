import warnings
warnings.filterwarnings("ignore")

# Standard Libraries
import os
import numpy as np
import pandas as pd

# Loading .mat files
import scipy.io

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Saving Images
import cv2

def rpca(
    M : np.ndarray,
    mu : float = 1,
    _lambda : float = 0.01,
    max_iteration : int = 100
) -> tuple[np.ndarray, np.ndarray]:
    '''
    '''
    S = np.zeros(M.shape)
    Y = np.zeros(M.shape)
    for i in range(max_iteration):
        X = M - S + Y / mu

        threshold = 1 / mu
        U, s, Vt = np.linalg.svd(X)
        s_tr = s[s > threshold]
        U_tr = U[:, :len(s_tr)]
        Vt_tr = Vt[:len(s_tr), :]
        L = U_tr @ np.diag(s_tr) @ Vt_tr

        X = M - L + Y / mu

        S = np.sign(X) * np.maximum(np.abs(X) - _lambda / mu, 0)

        Y += mu * (M - L - S)
    return L, S

if __name__ == '__main__':
    np.random.seed(1234)

    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'FinalQ4P1.mat'))

    M = scipy.io.loadmat(file_directory)['X']

    ## ====================================
    ## Part 1:
    ## ====================================
    L, S = rpca(M, 0.01)

    L = ((L - L.min()) / (L.max() - L.min())) * 255
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'L.png'))
    cv2.imwrite(file_directory, L)

    S = ((S - S.min()) / (S.max() - S.min())) * 255
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'S.png'))
    cv2.imwrite(file_directory, S)

    M = ((M - M.min()) / (M.max() - M.min())) * 255
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'M.png'))
    cv2.imwrite(file_directory, M)
    ## ====================================