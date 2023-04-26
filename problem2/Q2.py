import warnings
warnings.filterwarnings("ignore")

# Standard Libraries
import os
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    np.random.seed(1234)

    current_path = os.path.abspath(__file__)
    train_data_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Q2train.csv'))
    test_data_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Q2test.csv'))

    train_data = pd.read_csv(train_data_path, header = 0)
    test_data = pd.read_csv(test_data_path, header = 0)

    letter_to_number_mapping = {'s ': 1, 'h ': 2, 'd ': 3, 'o ': 4}

    X_train = train_data.iloc[:, 1:]
    X_test = test_data.iloc[:, 1:]

    y_train = train_data.iloc[:, 0].map(letter_to_number_mapping)
    y_test = test_data.iloc[:, 0].map(letter_to_number_mapping)

    ## =============================================
    ## Model 1: Logistic Regression
    ## =============================================
    model = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot = True, fmt = 'd')
    plt.title('Confusion Matrix: Logistic Regression')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    image_path = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'multinomial_logistic_regression_confusion_matrix.png'))
    plt.savefig(image_path)

    plt.clf()
    plt.cla()
    ## =============================================

    ## =============================================
    ## Model 2: Ridge Logistic Regression
    ## =============================================
    model = LogisticRegression(multi_class = 'multinomial',  solver = 'saga', penalty = 'l2', random_state = 42)

    param_grid = {'C': np.logspace(-4, 4, 20)}
    grid_search = GridSearchCV(model, param_grid, cv = 5, scoring = 'accuracy')
    grid_search.fit(X_train, y_train)

    optimal_C = grid_search.best_params_['C']
    print(f'''
##################################################    
Ridge logistic regression optimal tuning parameter (C):
{optimal_C}
##################################################
    ''')

    best_log_reg = grid_search.best_estimator_
    best_log_reg.fit(X_train, y_train)

    y_pred = best_log_reg.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot = True, fmt = 'd')
    plt.title('Confusion Matrix: Ridge Logistic Regression')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    image_path = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'multinomial_ridge_logistic_regression_confusion_matrix.png'))
    plt.savefig(image_path)

    plt.clf()
    plt.cla()
    ## =============================================

    ## =============================================
    ## Model 3: Lasso Logistic Regression
    ## =============================================
    model = LogisticRegression(multi_class = 'multinomial',  solver = 'saga', penalty = 'l1', random_state = 42)

    param_grid = {'C': np.logspace(-4, 4, 20)}
    grid_search = GridSearchCV(model, param_grid, cv = 5, scoring = 'accuracy')
    grid_search.fit(X_train, y_train)

    optimal_C = grid_search.best_params_['C']
    print(f'''
##################################################    
Lasso logistic regression optimal tuning parameter (C):
{optimal_C}
##################################################
    ''')

    best_log_reg = grid_search.best_estimator_
    best_log_reg.fit(X_train, y_train)

    y_pred = best_log_reg.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot = True, fmt = 'd')
    plt.title('Confusion Matrix: Lasso Logistic Regression')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    image_path = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'multinomial_lasso_logistic_regression_confusion_matrix.png'))
    plt.savefig(image_path)

    plt.clf()
    plt.cla()
    ## =============================================