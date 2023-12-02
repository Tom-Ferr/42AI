import matplotlib.pyplot as plt
import pandas as pd
import sys

try:
    thetas = pd.read_csv('.thetas.csv')
except:
    print('Pelase, run train.py first')
    exit(1)
try:
    data = pd.read_csv('data.csv')
    t0 = thetas.loc[0][0]
    t1 = thetas.loc[0][1]
    X = data.loc[0:]['km']
    Y = data.loc[0:]['price']
    Y_hat = X * t1 + t0
    Y_bar = Y.mean()
    n = Y.shape[0]
    sigma1 = 0  #((Y_hat - Y_bar) ** 2).sum()
    sigma2 = 0  #((Y - Y_bar) ** 2).sum()
    for i in range(n):
        sigma1 += (Y_hat.loc[i] - Y_bar) ** 2
        sigma2 += (Y.loc[i] - Y_bar) ** 2
    r_squared = sigma1 / sigma2
    print("The precision of the predictions is of: {}%".format(int(round(r_squared, 2) * 100)))
except:
    print('Please, make sure that data.csv was downloaded')
    exit(2)