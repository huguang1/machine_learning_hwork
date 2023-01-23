import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.svm import LinearSVC
import warnings

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None
pd.options.display.max_rows = None

df = pd.read_csv("dataset1.csv")
feature_1 = df.iloc[:, 0]  # first feature
feature_2 = df.iloc[:, 1]  # second feature
z = df.iloc[:, 2]  # result
features = np.column_stack((feature_1, feature_2))  # conbine feature to a array


# i(b)
def i_a1():
    """
    this function use cross_val_score to get f1 score
    use KFold to get parameters
    :return:
    """
    score_list = []
    data_dict = {}
    c_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
    for q in range(7):
        poly = PolynomialFeatures(degree=q + 1, include_bias=False)
        x_poly = poly.fit_transform(features)
        tempor_list = []
        for c in c_range:
            model = LogisticRegression(penalty='l2', C=c)
            scores = cross_val_score(model, x_poly, z, cv=5, scoring='f1')
            tempor_list.append(scores.mean())
        max_score = max(tempor_list)
        score_list.append((q + 1, max_score))
        tempor_list.insert(0, max_score)
        data_dict[q + 1] = tempor_list
    df = pd.DataFrame(data_dict, index=['max'] + [c for c in c_range])
    pd.set_option('max_colwidth', 100)
    print(df)


def i_a2():
    """
    this function is to plot mean_error and std_error
    :return:
    """
    poly = PolynomialFeatures(degree=3, include_bias=False)
    x_poly = poly.fit_transform(features)
    c_range = [10, 50, 100, 1000, 10000, 100000]
    mean_score, std_score = [], []
    for c in c_range:
        model = LogisticRegression(penalty='l2', C=c)
        scores = cross_val_score(model, x_poly, z, cv=5, scoring='f1')
        mean_score.append(np.array(scores).mean())
        std_score.append(np.array(scores).std())
    plt.axes(xscale="log")
    print(mean_score)
    plt.errorbar(c_range, mean_score, yerr=std_score)
    plt.xlabel('C')
    plt.ylabel('f1 score')
    plt.title("LogisticRegression with l2 and q=3")  # title
    plt.xlim((0, 120000))
    plt.show()


def i_a3():
    """
    this function is to plot data
    :return:
    """
    x, y = feature_1, feature_2
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', label='training data', alpha=0.5)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    x_poly = poly.fit_transform(features)
    Xtrain, Xtest, ytrain, ytest = train_test_split(x_poly, z, test_size=0.2)
    model = LogisticRegression(penalty='l2', C=1000).fit(Xtrain, ytrain)
    predict_z = model.predict(x_poly)
    ax.scatter(x, y, predict_z, c='g', label='predict data')
    plt.title("LOGISTIC REGRESSION")  # title
    ax.set_xlabel('first_feature')
    ax.set_ylabel('second_feature')
    ax.set_zlabel('target')
    plt.legend()
    plt.show()


def i_b():
    mean_, std_ = [], []
    for k in range(20):
        model = KNeighborsClassifier(n_neighbors=k + 1, weights='uniform')
        scores = cross_val_score(model, features, z, cv=5, scoring='f1')
        mean_.append(np.array(scores).mean())
        std_.append(np.array(scores).std())
    print(mean_)
    plt.errorbar([i + 1 for i in range(20)], mean_, yerr=std_)
    plt.xlabel('C')
    plt.ylabel('f1 score')
    plt.title("kNN")  # title
    plt.xlim((0, 20))
    plt.show()


def i_c():
    """
    this function use dummy model to calculate a baseline predictor
    :return:
    """
    dummy = DummyClassifier(strategy="most_frequent").fit(features, z)
    ydummy = dummy.predict(features)
    print(confusion_matrix(z, ydummy))
    print(classification_report(z, ydummy))

    dummy = DummyClassifier(strategy="uniform").fit(features, z)
    ydummy = dummy.predict(features)
    print(confusion_matrix(z, ydummy))
    print(classification_report(z, ydummy))

    Xtrain, Xtest, ytrain, ytest = train_test_split(features, z, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=7, weights='uniform').fit(Xtrain, ytrain)
    predict_z = model.predict(Xtest)
    print(confusion_matrix(ytest, predict_z))
    print(classification_report(ytest, predict_z))

    poly = PolynomialFeatures(degree=3, include_bias=False)
    x_poly = poly.fit_transform(features)
    Xtrain, Xtest, ytrain, ytest = train_test_split(x_poly, z, test_size=0.2)
    model = LogisticRegression(penalty='l2', C=1000).fit(Xtrain, ytrain)
    predict_z = model.predict(Xtest)
    print(confusion_matrix(ytest, predict_z))
    print(classification_report(ytest, predict_z))


def i_d():
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    Xtrain, Xtest, ytrain, ytest = train_test_split(features, z, test_size=0.2)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest, dummy.predict_proba(Xtest)[:, 1])
    plt.plot(fpr, tpr, c='b', label='baseline most_frequent', linestyle='-.')

    dummy = DummyClassifier(strategy="uniform").fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest, dummy.predict_proba(Xtest)[:, 1])
    plt.plot(fpr, tpr, c='y', label='baseline uniform', linestyle='--')

    model = KNeighborsClassifier(n_neighbors=7, weights='uniform').fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest, model.predict_proba(Xtest)[:, 1])
    plt.plot(fpr, tpr, c='r', label='kNN')

    poly = PolynomialFeatures(degree=3, include_bias=False)
    x_poly = poly.fit_transform(features)
    Xtrain, Xtest, ytrain, ytest = train_test_split(x_poly, z, test_size=0.2)
    model = LogisticRegression(penalty='l2', C=1000).fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest, model.decision_function(Xtest))
    plt.plot(fpr, tpr, c='g', label='Logistic Regression')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')
    plt.show()


def ii_a1():
    """
    this function use cross_val_score to get f1 score
    use KFold to get parameters
    :return:
    """
    score_list = []
    data_dict = {}
    c_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
    for q in range(10):
        poly = PolynomialFeatures(degree=q + 1, include_bias=False)
        x_poly = poly.fit_transform(features)
        tempor_list = []
        for c in c_range:
            model = LogisticRegression(penalty='l2', C=c)
            scores = cross_val_score(model, x_poly, z, cv=5, scoring='f1')
            tempor_list.append(scores.mean())
        max_score = max(tempor_list)
        score_list.append((q + 1, max_score))
        tempor_list.insert(0, max_score)
        data_dict[q + 1] = tempor_list
    df = pd.DataFrame(data_dict, index=['max'] + [c for c in c_range])
    pd.set_option('max_colwidth', 100)
    print(df)


def ii_a2():
    """
    this function is to plot mean_error and std_error
    :return:
    """
    poly = PolynomialFeatures(degree=7, include_bias=False)
    x_poly = poly.fit_transform(features)
    c_range = [0.001, 0.01, 0.1, 1, 2, 5, 10]
    mean_error, std_error = [], []
    for c in c_range:
        model = LogisticRegression(penalty='l2', C=c)
        scores = cross_val_score(model, x_poly, z, cv=5, scoring='f1')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.axes(xscale="log")
    print(mean_error)
    plt.errorbar(c_range, mean_error, yerr=std_error)
    plt.xlabel('C')
    plt.ylabel('f1 score')
    plt.title("LogisticRegression with l2 and q=7")  # title
    plt.xlim((0, 12))
    plt.show()


def ii_a3():
    """
    this function is to plot data
    :return:
    """
    x, y = feature_1, feature_2
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', label='training data', alpha=0.5)
    poly = PolynomialFeatures(degree=7, include_bias=False)
    x_poly = poly.fit_transform(features)
    Xtrain, Xtest, ytrain, ytest = train_test_split(x_poly, z, test_size=0.2)
    model = LogisticRegression(penalty='l2', C=1).fit(Xtrain, ytrain)
    predict_z = model.predict(x_poly)
    ax.scatter(x, y, predict_z, c='g', label='predict data')
    plt.title("LOGISTIC REGRESSION")  # title
    ax.set_xlabel('first_feature')
    ax.set_ylabel('second_feature')
    ax.set_zlabel('target')
    plt.legend()
    plt.show()


def ii_b():
    mean_, std_ = [], []
    m = 75
    for k in range(m):
        model = KNeighborsClassifier(n_neighbors=k + 1, weights='uniform')
        scores = cross_val_score(model, features, z, cv=5, scoring='f1')
        mean_.append(np.array(scores).mean())
        std_.append(np.array(scores).std())
    print(mean_)
    print(max(mean_))
    a = max(mean_)
    for index, value in enumerate(mean_):
        if value == a:
            print(index)
    plt.errorbar([i + 1 for i in range(m)], mean_, yerr=std_)
    plt.xlabel('C')
    plt.ylabel('f1 score')
    plt.title("kNN")  # title
    plt.xlim((0, m))
    plt.show()
    # k = 71的时候最大


def ii_c():
    """
    this function use dummy model to calculate a baseline predictor
    :return:
    """
    dummy = DummyClassifier(strategy="most_frequent").fit(features, z)
    ydummy = dummy.predict(features)
    print(confusion_matrix(z, ydummy))
    print(classification_report(z, ydummy))

    dummy = DummyClassifier(strategy="uniform").fit(features, z)
    ydummy = dummy.predict(features)
    print(confusion_matrix(z, ydummy))
    print(classification_report(z, ydummy))

    Xtrain, Xtest, ytrain, ytest = train_test_split(features, z, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=71, weights='uniform').fit(Xtrain, ytrain)
    predict_z = model.predict(Xtest)
    print(confusion_matrix(ytest, predict_z))
    print(classification_report(ytest, predict_z))

    poly = PolynomialFeatures(degree=7, include_bias=False)
    x_poly = poly.fit_transform(features)
    Xtrain, Xtest, ytrain, ytest = train_test_split(x_poly, z, test_size=0.2)
    model = LogisticRegression(penalty='l2', C=1).fit(Xtrain, ytrain)
    predict_z = model.predict(Xtest)
    print(confusion_matrix(ytest, predict_z))
    print(classification_report(ytest, predict_z))


def ii_d():
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    Xtrain, Xtest, ytrain, ytest = train_test_split(features, z, test_size=0.2)

    dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest, dummy.predict_proba(Xtest)[:, 1])
    plt.plot(fpr, tpr, c='b', label='baseline most_frequent', linestyle='-.')

    dummy = DummyClassifier(strategy="uniform").fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest, dummy.predict_proba(Xtest)[:, 1])
    plt.plot(fpr, tpr, c='y', label='baseline uniform', linestyle='--')

    model = KNeighborsClassifier(n_neighbors=71, weights='uniform').fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest, model.predict_proba(Xtest)[:, 1])
    plt.plot(fpr, tpr, c='r', label='kNN')
    poly = PolynomialFeatures(degree=7, include_bias=False)
    x_poly = poly.fit_transform(features)
    Xtrain, Xtest, ytrain, ytest = train_test_split(x_poly, z, test_size=0.2)
    model = LogisticRegression(penalty='l2', C=1).fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest, model.decision_function(Xtest))
    plt.plot(fpr, tpr, c='g', label='Logistic Regression')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    i_a2()
