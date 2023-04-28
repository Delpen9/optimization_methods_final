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

def construct_tensor(
    X : np.ndarray
) -> np.ndarray:
    '''
    '''
    n = X.shape[0]
    mat_1 = np.tile(np.arange(n), (n, 1)).T
    mat_2 = mat_1.T.copy()

    tensor = np.stack((mat_1, mat_2), axis = 2)
    return tensor

def linear_kernel_construction(
    tensor : np.ndarray,
    a : int = 1,
    b : int = 1
) -> np.ndarray:
    '''
    '''
    matrix = np.prod(tensor, axis = 2)
    gram_matrix = a + b * matrix
    return gram_matrix

def polynomial_kernel_construction(
    tensor : np.ndarray,
    c : int = 1,
    d : int = 10
) -> np.ndarray:
    '''
    '''
    matrix = np.prod(tensor, axis = 2)
    gram_matrix = (matrix + c)**d
    return gram_matrix

def periodic_kernel_construction(
    tensor : np.ndarray,
    e : int = 2,
    f : int = 5
) -> np.ndarray:
    '''
    '''
    matrix = tensor[:, :, 0] - tensor[:, :, 1]
    gram_matrix = np.exp(-2 * np.sin(np.pi * np.abs(matrix) / e)**2 / f)
    return gram_matrix

def gaussian_kernel_construction(
    tensor : np.ndarray,
    g : int = 0.04
) -> np.ndarray:
    '''
    '''
    matrix = tensor[:, :, 0] - tensor[:, :, 1]
    gram_matrix = np.exp(-(matrix)**2 / (2 * g))
    return gram_matrix

def gram_matrix(
    X : np.ndarray,
    kernel : str = 'linear'
)-> np.ndarray:
    '''
    '''
    tensor = construct_tensor(X)

    if kernel == 'linear':
        gram_matrix = linear_kernel_construction(tensor)
    elif kernel == 'polynomial':
        gram_matrix = polynomial_kernel_construction(tensor)
    elif kernel == 'periodic':
        gram_matrix = periodic_kernel_construction(tensor)
    else:
        gram_matrix = gaussian_kernel_construction(tensor)

    return gram_matrix

def RKHS(
    K : np.ndarray,
    y : np.ndarray
) -> np.ndarray:
    '''
    '''
    _lambda = 0.1
    _alpha = np.linalg.inv(K + _lambda * np.eye(K.shape[0])) @ y
    y_hat = K @ _alpha
    return y_hat

def RPCA(
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

    ## ====================================
    ## Part 1:
    ## ====================================
    # file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'FinalQ4P1.mat'))
    # M = scipy.io.loadmat(file_directory)['X']

    # L, S = RPCA(M, 0.01)

    # L = ((L - L.min()) / (L.max() - L.min())) * 255
    # file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'L.png'))
    # cv2.imwrite(file_directory, L)

    # S = ((S - S.min()) / (S.max() - S.min())) * 255
    # file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'S.png'))
    # cv2.imwrite(file_directory, S)

    # M = ((M - M.min()) / (M.max() - M.min())) * 255
    # file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'M.png'))
    # cv2.imwrite(file_directory, M)
    ## ====================================

    ## ====================================
    ## Part 2.1: 
    ## ====================================
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'FinalQ4P2Train.csv'))
    data_train = np.genfromtxt(file_directory, delimiter = ',')

    X_train = data_train[:, [0]].copy()
    y_train = data_train[:, [1]].flatten().copy()
    
    linear_kernel = gram_matrix(X_train, 'linear')
    polynomial_kernel = gram_matrix(X_train, 'polynomial')
    periodic_kernel = gram_matrix(X_train, 'periodic')
    gaussian_kernel = gram_matrix(X_train, 'gaussian')

    linear_kernel = ((linear_kernel - linear_kernel.min()) / (linear_kernel.max() - linear_kernel.min())) * 255
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'linear_kernel.png'))
    cv2.imwrite(file_directory, linear_kernel)

    polynomial_kernel = ((polynomial_kernel - polynomial_kernel.min()) / (polynomial_kernel.max() - polynomial_kernel.min())) * 255
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'polynomial_kernel.png'))
    cv2.imwrite(file_directory, polynomial_kernel)

    periodic_kernel = ((periodic_kernel - periodic_kernel.min()) / (periodic_kernel.max() - periodic_kernel.min())) * 255
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'periodic_kernel.png'))
    cv2.imwrite(file_directory, periodic_kernel)

    gaussian_kernel = ((gaussian_kernel - gaussian_kernel.min()) / (gaussian_kernel.max() - gaussian_kernel.min())) * 255
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'gaussian_kernel.png'))
    cv2.imwrite(file_directory, gaussian_kernel)
    ## ====================================

    ## ====================================
    ## Part 2.2:
    ## NOTE: X_train and X_test are identical
    ## ====================================
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'FinalQ4P2Test.csv'))
    data_test = np.genfromtxt(file_directory, delimiter = ',')

    X_test = data_test[:, [0]].copy()
    y_test = data_test[:, [1]].flatten().copy()

    linear_kernel = gram_matrix(X_train, 'linear')
    polynomial_kernel = gram_matrix(X_train, 'polynomial')
    periodic_kernel = gram_matrix(X_train, 'periodic')
    gaussian_kernel = gram_matrix(X_train, 'gaussian')
    
    print('#####################################################')
    y_pred_linear = RKHS(linear_kernel, y_train)
    rmse = np.sqrt(np.mean((y_pred_linear - y_test) ** 2))
    print(f'''The RMSE on the test set for the linear kernel is: \n{round(rmse, 6)}''')

    y_pred_polynomial = RKHS(polynomial_kernel, y_train)
    rmse = np.sqrt(np.mean((y_pred_polynomial - y_test) ** 2))
    print(f'''The RMSE on the test set for the polynomial kernel is: \n{round(rmse, 6)}''')

    y_pred_periodic = RKHS(periodic_kernel, y_train)
    rmse = np.sqrt(np.mean((y_pred_periodic - y_test) ** 2))
    print(f'''The RMSE on the test set for the periodic kernel is: \n{round(rmse, 6)}''')

    y_pred_gaussian = RKHS(gaussian_kernel, y_train)
    rmse = np.sqrt(np.mean((y_pred_gaussian - y_test) ** 2))
    print(f'''The RMSE on the test set for the gaussian kernel is: \n{round(rmse, 6)}''')
    print('#####################################################')

    prediction_df = pd.DataFrame(np.hstack((
        X_train.reshape(-1, 1),
        y_pred_linear.reshape(-1, 1),
        y_pred_polynomial.reshape(-1, 1),
        y_pred_periodic.reshape(-1, 1),
        y_pred_gaussian.reshape(-1, 1),
        y_test.reshape(-1, 1),
    )), columns = ['X', 'y_pred_linear', 'y_pred_polynomial', 'y_pred_periodic', 'y_pred_gaussian', 'y_test'])

    ## ====================================
    sns.lineplot(data = prediction_df, x = 'X', y = 'y_pred_linear')
    sns.lineplot(data = prediction_df, x = 'X', y = 'y_test')
    plt.title('Linear Kernel Y Predictions VS. True Y')
    plt.xlabel('X')
    plt.ylabel('Y')

    image_path = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'q4_linear_kernel_vs_true_y.png'))
    plt.savefig(image_path)

    plt.clf()
    plt.cla()
    ## ====================================

    ## ====================================
    sns.lineplot(data = prediction_df, x = 'X', y = 'y_pred_polynomial')
    sns.lineplot(data = prediction_df, x = 'X', y = 'y_test')
    plt.title('Polynomial Kernel Y Predictions VS. True Y')
    plt.xlabel('X')
    plt.ylabel('Y')

    image_path = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'q4_polynomial_kernel_vs_true_y.png'))
    plt.savefig(image_path)

    plt.clf()
    plt.cla()
    ## ====================================

    ## ====================================
    sns.lineplot(data = prediction_df, x = 'X', y = 'y_pred_periodic')
    sns.lineplot(data = prediction_df, x = 'X', y = 'y_test')
    plt.title('Periodic Kernel Y Predictions VS. True Y')
    plt.xlabel('X')
    plt.ylabel('Y')

    image_path = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'q4_periodic_kernel_vs_true_y.png'))
    plt.savefig(image_path)

    plt.clf()
    plt.cla()
    ## ====================================

    ## ====================================
    sns.lineplot(data = prediction_df, x = 'X', y = 'y_pred_gaussian')
    sns.lineplot(data = prediction_df, x = 'X', y = 'y_test')
    plt.title('Gaussian Kernel Y Predictions VS. True Y')
    plt.xlabel('X')
    plt.ylabel('Y')

    image_path = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'q4_gaussian_kernel_vs_true_y.png'))
    plt.savefig(image_path)

    plt.clf()
    plt.cla()
    ## ====================================

    ## ====================================