import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import svm
import math
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv("week3.csv")
feature_1 = df.iloc[:, 0]  # first feature
feature_2 = df.iloc[:, 1]  # second feature
z = df.iloc[:, 2]  # result
features = np.column_stack((feature_1, feature_2))  # conbine feature to a array


# i(a)
def i_a():
    x, y = feature_1, feature_2,
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('first_feature')
    ax.set_ylabel('second_feature')
    ax.set_zlabel('target')
    plt.show()
    plt.show()


if __name__ == '__main__':
    i_a()





