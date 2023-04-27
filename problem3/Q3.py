import warnings
warnings.filterwarnings("ignore")

# Standard Libraries
import os
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

def relative_reconstruction_error(
    Z : np.ndarray,
    X0 : np.ndarray
) -> float:
    '''
    '''
    epsilon = np.linalg.norm(Z - X0, ord = 'fro') / np.linalg.norm(X0, ord = 'fro')
    return epsilon

def PFBS(
    X0 : np.ndarray,
    X : np.ndarray,
    A : np.ndarray,
    m : int,
    n1 : int,
    n2 : int,
    tau : int = 200,
    max_iteration : int = 200
) -> np.ndarray:
    '''
    '''
    Y = np.zeros((n1, n2))
    delta = 0.1
    for i in range(max_iteration):
        U, S, V = np.linalg.svd(Y)
        S_diag = np.zeros(Y.shape)
        S_diag[np.diag_indices(Y.shape[0])] = S

        S_t = (S_diag - tau)
        S_t[S_t < 0] = 0

        Z = U @ S_t @ V
        P = X - Z
        P[A] = 0

        Y0 = Y.copy()
        Y = Y0 + delta * P
    return np.round(Z, decimals = 0).astype(int)

if __name__ == '__main__':
    np.random.seed(1234)

    current_path = os.path.abspath(__file__)
    data_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'ratings.csv'))

    # Part 1
    zero_rates = np.arange(0.05, 0.55, 0.05)
    reconstruction_errors = []
    for zero_rate in zero_rates:
        R = np.genfromtxt(data_path, delimiter = ',')
        R0 = R.copy()

        A = np.random.rand(*R.shape) >= (1 - zero_rate)
        R[A] = 0
        m = np.sum(np.sum(R == 0))

        Z = PFBS(R0, R, A, m, *R.shape)

        reconstruction_errors.append(relative_reconstruction_error(Z, R0))

        if zero_rate == 0.5:
            print('#####################################################')
            print('The matrix with missing values:')
            unique_values, counts = np.unique(R.astype(int).flatten(), return_counts = True)
            for value, count in zip(unique_values, counts):
                print(f'{value} appears {count} times.')
            print('#####################################################')
            print('The recovered matrix:')
            unique_values, counts = np.unique(Z.flatten(), return_counts = True)
            for value, count in zip(unique_values, counts):
                print(f'{value} appears {count} times.')
            print('#####################################################')
            print('The original matrix:')
            unique_values, counts = np.unique(R0.astype(int).flatten(), return_counts = True)
            for value, count in zip(unique_values, counts):
                print(f'{value} appears {count} times.')
            print('#####################################################')

    # ===================================
    # Plot 1: Part 1 - Reconstruction Error VS. Percentage of Missing Data
    # ===================================
    percent_of_missing_data = zero_rates.reshape(-1, 1) * 100
    reconstruction_errors = np.array(reconstruction_errors).reshape(-1, 1)

    plot_np = np.hstack((percent_of_missing_data, reconstruction_errors))
    plot_df = pd.DataFrame(plot_np, columns = ['Percent of Missing Data', 'Reconstruction Error'])

    sns.lineplot(data = plot_df, x = 'Percent of Missing Data', y = 'Reconstruction Error')
    plt.title('Part 1: Percent of Missing Data VS. Relative Reconstruction Error')
    plt.xlabel('Percent of Missing Data (%)')
    plt.ylabel('Relative Reconstruction Error')

    image_path = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'q3_part1_missing_data_vs_reconstruction_error.png'))
    plt.savefig(image_path)

    plt.clf()
    plt.cla()
    # ===================================

    # Part 2
    zero_rates = np.arange(0.05, 0.55, 0.05)
    reconstruction_errors = []
    for zero_rate in zero_rates:
        R = np.genfromtxt(data_path, delimiter = ',')
        R0 = R.copy()

        unique_values, counts = np.unique(np.sort(R0.astype(int).flatten()), return_counts = True)
        count_12 = counts[0] + counts[1]
        count_34 = counts[2] + counts[3]
        counts_12345 = sum(counts)

        prob_X = zero_rate * counts_12345 / (3 * count_12 + count_34)

        A = np.zeros(R0.shape, dtype = bool)

        mask_12 = np.random.rand(*R0.shape)
        condition = ((R0 == 1) | (R0 == 2))
        A[condition & (mask_12 >= 1 - 3 * prob_X)] = True

        mask_34 = np.random.rand(*R0.shape)
        condition = ((R0 == 3) | (R0 == 4))
        A[condition & (mask_34 >= 1 - prob_X)] = True

        R[A] = 0
        m = np.sum(np.sum(R == 0))

        Z = PFBS(R0, R, A, m, *R.shape)

        reconstruction_errors.append(relative_reconstruction_error(Z, R0))

        if zero_rate == 0.5:
            print('#####################################################')
            print('The matrix with missing values:')
            unique_values, counts = np.unique(R.astype(int).flatten(), return_counts = True)
            for value, count in zip(unique_values, counts):
                print(f'{value} appears {count} times.')
            print('#####################################################')
            print('The recovered matrix:')
            unique_values, counts = np.unique(Z.flatten(), return_counts = True)
            for value, count in zip(unique_values, counts):
                print(f'{value} appears {count} times.')
            print('#####################################################')
            print('The original matrix:')
            unique_values, counts = np.unique(R0.astype(int).flatten(), return_counts = True)
            for value, count in zip(unique_values, counts):
                print(f'{value} appears {count} times.')
            print('#####################################################')

    # ===================================
    # Plot 2: Part 2 - Reconstruction Error VS. Percentage of Missing Data
    # ===================================
    percent_of_missing_data = zero_rates.reshape(-1, 1) * 100
    reconstruction_errors = np.array(reconstruction_errors).reshape(-1, 1)

    plot_np = np.hstack((percent_of_missing_data, reconstruction_errors))
    plot_df = pd.DataFrame(plot_np, columns = ['Percent of Missing Data', 'Reconstruction Error'])

    sns.lineplot(data = plot_df, x = 'Percent of Missing Data', y = 'Reconstruction Error')
    plt.title('Part 2: Percent of Missing Data VS. Relative Reconstruction Error')
    plt.xlabel('Percent of Missing Data (%)')
    plt.ylabel('Relative Reconstruction Error')

    image_path = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'q3_part2_missing_data_vs_reconstruction_error.png'))
    plt.savefig(image_path)

    plt.clf()
    plt.cla()
    # ===================================W