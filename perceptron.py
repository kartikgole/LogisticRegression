#Karteek Gole 1001553522
#Sources (1): https://www.kaggle.com/jeppbautista/logistic-regression-from-scratch-python
#Sources (2): https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
#Sources (3): https://github.com/slntRohit/machine-learning/blob/master/logistic-regression/logistic-regression.py
#Sources (4): https://github.com/yatharth1908/Machine-Learning/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.optimize import minimize

mean = [1, 0]
mean1 = [0, 1.5]

cv = [[1, 0.75], [0.75, 1]]
cv1 = [[1, 0.75], [0.75, 1]]


def traindata(mean_1, mean_2, cv, cv1):
    #generating 1000 data points using random_multivariate_normal
    traindata_1 = np.append(np.random.multivariate_normal(mean, cv, 1000), np.zeros((1000, 1)), axis=1)
    traindata_2 = np.append(np.random.multivariate_normal(mean1, cv1, 1000), np.ones((1000, 1)), axis=1)
    train_data = np.concatenate((traindata_1, traindata_2))
    return train_data


def testdata(mean, mean1, cv, cv1):
    testdata_1 = np.append(np.random.multivariate_normal(mean, cv, 500), np.zeros((500, 1)), axis=1)
    testdata_2 = np.append(np.random.multivariate_normal(mean1, cv1, 500), np.ones((500, 1)), axis=1)
    test_data = np.concatenate((testdata_1, testdata_2))
    return test_data


data = traindata(mean, mean1, cv, cv1)
data_test = testdata(mean, mean1, cv, cv1)


def plotData(data, labelx, labely, positivelabel, negativelabel, axes=None):
    # Get indexes for class 0 and class 1
    negative = data[:, 2] == 0
    positive = data[:, 2] == 1

    # If no specific axes object has been passed, get the current axes.
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[positive][:, 0], data[positive][:, 1], marker='+', c='g', s=60, linewidth=2, label=positivelabel)
    axes.scatter(data[negative][:, 0], data[negative][:, 1], c='y', s=60, label=negativelabel)
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    axes.legend(frameon=True, fancybox=True)
    plt.show()


X = np.c_[np.ones((data.shape[0], 1)), data[:, 0:2]]
y = np.c_[data[:, 2]]

X_test = np.c_[np.ones((data_test.shape[0], 1)), data_test[:, 0:2]]
y_test = np.c_[data_test[:, 2]]


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

#activation using sigmoid
def predict(wts, X, learningrate):
    #print(theta)
    p = sigmoid(X.dot(wts.T)) >= learningrate
    return (p.astype('int'))

def gradient(wts, X, y):
    m = y.size
    h = sigmoid(X.dot(wts.reshape(-1, 1)))
    grad = (1 / m) * X.T.dot(h - y)
    gradflat = grad.flatten()
    return (gradflat)

# ojective function cross-entropy
def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))

    # dividing derivative by m(size of training set) to avoid NaN

    newtheta = -1 * (1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))

    #rechecking to avoid NaN
    if np.isnan(newtheta[0]):
        return (np.inf)

    #print(J[0])
    return (newtheta[0])


def pltroc(y_test, p):
    false, true, threshold = roc_curve(y_test, p)
    roc_auc = auc(false, true)

    plt.title('ROC Curve')
    plt.plot(false, true, label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    x = false[1]

    plt.fill(false, true, 'c', true, false, 'c', [0.0, 1.0, 1.0], [0.0, 0.0, x], 'c')
    plt.show()


# initial weights [ 1. 1. 1.]
init_wts = np.ones(X.shape[1])

#print(initial_theta)

cost = costFunction(init_wts, X, y)
grad = gradient(init_wts, X, y)

print('Weights: \n', cost)
print('Gradient: \n', grad)


#based on batch training, executing entropy cost function 9999 times with initial value of theta = [1. 1. 1.]
res = minimize(costFunction, init_wts, args=(X, y), options={'maxiter': 9999})
#print(res.x)


#Online training of data based on the latest grad value retrieved from above
resnew = minimize(costFunction, grad, args=(X, y), options={'maxiter': 10000})
#print(resnew.x)

#predictions based with batch training
prediction = predict(res.x, X_test, 0.1)
prediction1 = predict(res.x, X_test, 0.01)
prediction2 = predict(res.x, X_test, 1)

#predictions based with online training
prediction4online = predict(resnew.x, X_test, 0.1)
prediction4online1 = predict(resnew.x, X_test, 0.01)
prediction4online2 = predict(resnew.x, X_test, 1)

#for the piplot
x1_min, x1_max = X_test[:, 1].min(), X_test[:, 1].max()
x2_min, x2_max = X_test[:, 2].min(), X_test[:, 2].max()

xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)

plt.contour(xx1, xx2, h, [0.5], linewidths=2, colors='k')

#Accuracy without online training
print('(Accuracy for lr = 0.1) {}%'.format(100 * sum(prediction == y_test.ravel()) / prediction.size))
print('(Accuracy for lr = 0.01) {}%'.format(100 * sum(prediction1 == y_test.ravel()) / prediction1.size))
print('(Accuracy for lr = 1) {}%'.format(100 * sum(prediction2 == y_test.ravel()) / prediction2.size))

#Accuracy with online training
print('(Accuracy for Online training lr = 0.1) {}%'.format(100 * sum(prediction4online == y_test.ravel()) / prediction4online.size))
print('(Accuracy for Online training lr = 0.01) {}%'.format(100 * sum(prediction4online1 == y_test.ravel()) / prediction4online1.size))
print('(Accuracy for Online training lr = 1) {}%'.format(100 * sum(prediction4online2 == y_test.ravel()) / prediction4online2.size))
print("Edges learned with batch training:- ")
print(res.x)
print("Edges learned with Online training:- ")
print(resnew.x)
pltroc(y_test, prediction)
pltroc(y_test, prediction1)
pltroc(y_test, prediction2)