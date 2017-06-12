import numpy as np

from datetime import datetime

from newton_raphson import newton_raphson
from newton_raphson import newton_raphson_vectorized

X = np.genfromtxt('logistic_x.txt')
y = np.genfromtxt('logistic_y.txt')

size = X.shape
mm = size[0] # number of samples
nn = size[1] # number of features

ones = np.ones((mm,1))

X = np.hstack((ones,X))
theta = np.zeros(nn + 1)

startTime = datetime.now()
theta2 = newton_raphson(X,y,theta,20)
print(datetime.now() - startTime)

startTime = datetime.now()
theta3 = newton_raphson_vectorized(X,y,theta,20)
print(datetime.now() - startTime)

diff = theta2 - theta3

print(diff)