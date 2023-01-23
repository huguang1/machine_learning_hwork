import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import svm
import math
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

""" 
this part is the common part, every function needs this part
"""
df = pd.read_csv("week2.csv")
feature_1 = df.iloc[:, 0]  # first feature
feature_2 = df.iloc[:, 1]  # second feature
y = df.iloc[:, 2]  # result
features = np.column_stack((feature_1, feature_2))  # conbine feature to a array


# a(i)
def a_i():
    """
    this part is for plot original data
    :return:
    """
    positive = np.array(y>0)  # this array let me use two type of point to show the data
    negative = np.array(y<0)
    plt.scatter(feature_1*positive, feature_2*positive, c='green', marker='+', label='+1 points')  # use + to replacement a point
    plt.scatter(feature_1*negative, feature_2*negative, c='blue', marker='o', label='-1 points')  # use + to replacement a point
    plt.title("ASSIONMENT LOGISTIC REGRESSION")  # title
    plt.xlabel("first_feature")  # x-axis
    plt.ylabel("second_feature")  # y-axis
    plt.legend(loc=0, bbox_to_anchor=(1, 1))
    plt.show()

# a(ii)
def  a_ii():
    """
    this function use LogisticRegression to train data
    :return:
    """
    model = LogisticRegression(penalty='none',solver='lbfgs')
    model.fit(features, y)
    print(model.coef_)
    print(model.intercept_)
    print(model.score(features, y))

# a(iii)
def a_iii():
    """
    this function put predicted data and original data together to draw a picture
    :return:
    """
    positive = np.array(y>0)
    negative = np.array(y<0)
    plt.scatter(feature_1*positive, feature_2*positive, c='green', marker='o', alpha=0.7, label='real +1')  # use + to replacement a point
    plt.scatter(feature_1*negative, feature_2*negative, c='blue', marker='o', alpha=0.7, label='real -1')  # use + to replacement a point
    prediction_positive = np.array((-1.77044152 + 0.14101238*feature_1 - 5.76025623*feature_2)>0)
    prediction_negative = np.array((-1.77044152 + 0.14101238*feature_1 - 5.76025623*feature_2)<0)
    plt.scatter(feature_1*prediction_positive, feature_2*prediction_positive, c='yellow', marker='+', label='predict +1')  # use + to replacement a point
    plt.scatter(feature_1*prediction_negative, feature_2*prediction_negative, c='red', marker='+', label='predict -1')  # use + to replacement a point
    x = np.linspace(-1, 1)
    y_predict = 0.02448023 * x - 0.30735465
    plt.plot(x, y_predict, 'k', label='predict line')  # Boundary line
    plt.title("ASSIONMENT LOGISTIC REGRESSION")  # title
    plt.xlabel("first_feature")  # x-axis
    plt.ylabel("second_feature")  # y-axis
    plt.legend(loc=0, bbox_to_anchor=(1, 1))
    plt.show()

# b(i)
def b_i():
    """
    this funciton use different c to train LinearSVC model.
    :return:
    """
    for c in [0.001, 1, 100]:
        print(c)
        model = svm.LinearSVC(C=c)
        model.fit(features, y)
        print(model.coef_)
        print(model.intercept_)
        print(model.score(features, y))

# b(ii)
def b_ii(C):
    """
    this funciotn use different parameter to plot picture
    :param C: choose different parameter to plot picture
    :return:
    """
    c_dict = {
        "C0": (-0.20688338, -0.00180362, -0.48161875),
        "C1": (-0.56865682, 0.03652591, -1.8778363),
        "C100": (-0.55309646, 0.05334568, -1.8426769)
    }
    positive = np.array(y>0)
    negative = np.array(y<0)
    plt.scatter(feature_1*positive, feature_2*positive, c='green', marker='o', alpha=0.7, label='real +1')  # use + to replacement a point
    plt.scatter(feature_1*negative, feature_2*negative, c='blue', marker='o', alpha=0.7, label='real -1')  # use + to replacement a point
    prediction_positive = np.array((c_dict[C][0] + c_dict[C][1]*feature_1 + c_dict[C][2]*feature_2)>0)
    prediction_negative = np.array((c_dict[C][0] + c_dict[C][1]*feature_1 + c_dict[C][2]*feature_2)<0)
    plt.scatter(feature_1*prediction_positive, feature_2*prediction_positive, c='yellow', marker='+', label='predict +1')  # use + to replacement a point
    plt.scatter(feature_1*prediction_negative, feature_2*prediction_negative, c='red', marker='+', label='predict -1')  # use + to replacement a point
    x = np.linspace(-1, 1)
    print(-c_dict[C][1] / c_dict[C][2])
    print(-c_dict[C][0]/c_dict[C][2])
    y_predict = - c_dict[C][1] / c_dict[C][2] * x - c_dict[C][0]/c_dict[C][2]
    plt.plot(x, y_predict, 'k', label='predict line')  # Boundary line
    plt.title("ASSIONMENT LOGISTIC REGRESSION")  # title
    plt.xlabel("first_feature")  # x-axis
    plt.ylabel("second_feature")  # y-axis
    plt.legend(loc=0, bbox_to_anchor=(1, 1))
    plt.show()

# c(i)
def c_i():
    """
    this function is use LogisticRegression to train data
    :return:
    """
    features = np.column_stack((feature_1, feature_2, feature_1**2, feature_2**2))  # conbine feature to a array
    model = LogisticRegression(penalty='none',solver='lbfgs')
    model.fit(features, y)
    print(model.coef_)
    print(model.intercept_)
    print(model.score(features, y))

# c(ii)
def c_ii():
    """
    this function put predicted data and original data together to draw a picture
    :return:
    """
    positive = np.array(y>0)
    negative = np.array(y<0)
    plt.scatter(feature_1*positive, feature_2*positive, c='green', marker='o', alpha=0.7, label='real +1')  # use + to replacement a point
    plt.scatter(feature_1*negative, feature_2*negative, c='blue', marker='o', alpha=0.7, label='real -1')  # use + to replacement a point
    prediction_positive = np.array((0.21213532 + 0.69995165*feature_1 - 24.75796056*feature_2 - 27.82103092*feature_1**2 + 3.25262485*feature_2**2)>0)
    prediction_negative = np.array((0.21213532 + 0.69995165*feature_1 - 24.75796056*feature_2 - 27.82103092*feature_1**2 + 3.25262485*feature_2**2)<0)
    plt.scatter(feature_1*prediction_positive, feature_2*prediction_positive, c='yellow', marker='+', label='predict +1')  # use + to replacement a point
    plt.scatter(feature_1*prediction_negative, feature_2*prediction_negative, c='red', marker='+', label='predict -1')  # use + to replacement a point
    plt.title("ASSIONMENT LOGISTIC REGRESSION")  # title
    plt.xlabel("first_feature")  # x-axis
    plt.ylabel("second_feature")  # y-axis
    plt.legend(loc=0, bbox_to_anchor=(1, 1))
    plt.show()

def c_iii():
    """
    this function use dummy model to calculate a baseline predictor
    :return:
    """
    dummy = DummyClassifier(strategy="most_frequent").fit(features, y)
    ydummy = dummy.predict(features)
    print(confusion_matrix(y, ydummy))
    print(classification_report(y, ydummy))

def c_iv():
    """
    this function is to draw a curve line
    :return:
    """
    positive = np.array(y>0)
    negative = np.array(y<0)
    plt.scatter(feature_1*positive, feature_2*positive, c='green', marker='o', alpha=0.7, label='real +1')  # use + to replacement a point
    plt.scatter(feature_1*negative, feature_2*negative, c='blue', marker='o', alpha=0.7, label='real -1')  # use + to replacement a point
    prediction_positive = np.array((0.21213532 + 0.69995165*feature_1 - 24.75796056*feature_2 - 27.82103092*feature_1**2 + 3.25262485*feature_2**2)>0)
    prediction_negative = np.array((0.21213532 + 0.69995165*feature_1 - 24.75796056*feature_2 - 27.82103092*feature_1**2 + 3.25262485*feature_2**2)<0)
    plt.scatter(feature_1*prediction_positive, feature_2*prediction_positive, c='yellow', marker='+', label='predict +1')  # use + to replacement a point
    plt.scatter(feature_1*prediction_negative, feature_2*prediction_negative, c='red', marker='+', label='predict -1')  # use + to replacement a point
    x = np.linspace(-1, 1)
    C = [0.21213532, 0.69995165, -24.75796056, -27.82103092, 3.25262485]  # function parameter
    def f(x):
        return -math.sqrt((C[2]/(2*C[4]))**2 - (C[0]+C[1]*x+C[3]*x**2)/C[4]) - C[2]/(2*C[4])
    f2 = np.vectorize(f)
    plt.plot(x, f2(x), 'k', label='predict line')  # Boundary line
    plt.title("ASSIONMENT LOGISTIC REGRESSION")  # title
    plt.xlabel("first_feature")  # x-axis
    plt.ylabel("second_feature")  # y-axis
    plt.legend(loc=0, bbox_to_anchor=(1, 1))
    plt.show()


if __name__ == '__main__':
    c_i()








































