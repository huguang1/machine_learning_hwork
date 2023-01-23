from sklearn.neural_network import MLPClassifier, MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

X1 = np.linspace(0.0, 1.0, num=10000)
X2 = np.linspace(0.0, 1.0, num=10000)
y = np.sign(X1 + X2 - 1 + np.random.normal(0, 0.4, 10000))
X1 = X1.reshape(-1, 1)
X2 = X2.reshape(-1, 1)
X = np.column_stack((X1, X2))

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
model = MLPClassifier(hidden_layer_sizes=(5), alpha=1.0 / 5).fit(Xtrain, ytrain)
preds = model.predict(Xtest)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(ytest, preds))
from sklearn.metrics import roc_curve

preds = model.predict_proba(Xtest)
print(model.classes_)
fpr, tpr, _ = roc_curve(ytest, preds[:, 1])
plt.plot(fpr, tpr, color='blue')
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1).fit(Xtrain, ytrain)
fpr, tpr, _ = roc_curve(ytest, model.decision_function(Xtest))
plt.plot(fpr, tpr, color='orange')
plt.legend(['MLP', 'Logistic Regression'])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.plot([0, 1], [0, 1], color='green', linestyle='-')
plt.show()
