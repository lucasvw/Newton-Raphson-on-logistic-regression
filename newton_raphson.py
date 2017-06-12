import numpy as np

def sigmoid(x):
    return np.divide(1.0, 1.0 + np.exp(-x))


def newton_raphson_vectorized(x, y, theta, max_iterations = 20 ):
    """This is a vectorized implementation of the newton raphson method"""

    mm = x.shape[0]

    for iteration in range(0, max_iterations):
        arg = (-1 * y) * x.dot(theta)
        probs = sigmoid(arg)
        J_diff_1 = (-1 / mm)*x.T.dot((probs*y))
        J_diff_2 = (1/mm)*(x.transpose().dot(np.diag((probs)*(1-probs))).dot(x))
        theta = theta - np.asarray(np.linalg.inv(np.mat(J_diff_2))).dot(J_diff_1)

    return theta




def newton_raphson(x, y, theta, max_iterations = 20 ):
    """This is an unvectorized implementation of the newton raphson method"""

    mm = x.shape[0]
    nn = x.shape[1]

    for iteration in range(0, max_iterations):
        J_diff_1 = np.zeros(nn)
        J_diff_2 = np.zeros((nn, nn))
        temp1 = np.zeros((mm, 1))
        temp2 = np.zeros((mm, 1))
        probs = np.zeros((mm, 1))

        for i in range(0, nn):
            for j in range(0,mm):
                probs[j] = sigmoid(-y[j] * np.dot(x[j, :], theta))
                temp1[j] = probs[j]*y[j] * x[j, i]
            J_diff_1[i] = (-1 / mm) * np.sum(temp1)

        for i in range(0,nn):
            for k in range(0,nn):
                for j in range(0,mm):
                    temp2[j] = probs[j] * (1 - probs[j]) * x[j, k] * x[j, i]
                    J_diff_2[i, k] = (1 / mm) * np.sum(temp2)

        theta = theta - np.asarray(np.linalg.inv(np.mat(J_diff_2))).dot(J_diff_1)

    return theta
