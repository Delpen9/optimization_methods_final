# Standard Libraries
import os
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

if __name__ == '__main___':
    
    X_train = None
    X_test = None

    y_train = None
    y_test = None

    model = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)