import matplotlib.pyplot as plt
import pandas as pd

def normalization(value, mini, maxi):
    return (value - mini) / (maxi -  mini)

def linear_interpolation(value, mini, maxi):
    return (maxi - mini) * value + mini

def derivatives(X, Y):
    tmp0 = tmp1 = 0
    for i in range(m):
        epsilon = theta_one * X[i] + theta_not - Y[i]
        tmp0 += epsilon
        tmp1 += epsilon * X[i]
    return tmp0 / m, tmp1 / m

def gradient_descent():
    global theta_not, theta_one
    theta_not = theta_one = 0
    X_norm = normalization(X, X.min(), X.max())
    Y_norm = normalization(Y, Y.min(), Y.max())
    for it in range(iters):
        print("\r\033[0Klearning... {}%".format(int((it+1) / iters * 100)), end="")
        tmp0, tmp1 = derivatives(X_norm, Y_norm)
        theta_not -= tmp0 * learning_rate
        theta_one -= tmp1 * learning_rate
    print("\n", end="")
    Y_hat_norm = theta_not + theta_one * X_norm
    Y_hat = linear_interpolation(Y_hat_norm, Y.min(), Y.max())              #prediction
    theta_one = (Y_hat[1] - Y_hat[0]) / (X[1] - X[0])                       #slope
    theta_not = Y_hat[0] - theta_one * X[0]                                 #interception
    print("Theta 0 = {}, Theta 1 = {}".format(theta_not, theta_one))

"""
Main
"""
#variables
data = pd.read_csv('data.csv')
m = data.shape[0]
learning_rate = 0.1
iters = 2000
theta_not = 0
theta_one = 0
X = data.loc[0:]['km']
Y = data.loc[0:]['price']

#run
gradient_descent()
data.plot(kind = 'scatter', x = 'km', y = 'price')
plt.show()
