import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data.csv')
m = data.shape[0]
theta_not = theta_one = 0
learning_rate = 0.0001

def tmp_theta(subscript: bool):
    sigma = 0
    for i in range(m):
        partial = (theta_not + (theta_one * data.loc[i][0])) - data.loc[i][1]
        if subscript:
            partial *= data.loc[i][0]
        sigma += partial
    return learning_rate * 1/m * sigma

data.plot(kind = 'scatter', x = 'km', y = 'price')
plt.show()