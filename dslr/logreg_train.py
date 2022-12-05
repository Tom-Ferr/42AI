import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.model_selection import train_test_split

def normalization(value, mini, maxi):
    return (value - mini) / (maxi -  mini)

def linear_interpolation(value, mini, maxi):
    return (maxi - mini) * value + mini

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def hypothesis(x, thetas):
    return sigmoid(np.dot(x, thetas))

def gradient_descent(X,Y):
    learning_rate = 0.1
    epoch = 2000
    x = np.hstack((np.ones((X.shape[0],1)), X))
    m = x.shape[0]
    theta = []
    
    count = 0
    for house in np.unique(Y):
        thetas = np.zeros(x.shape[1])
        y = np.where(Y == house, 1, 0)
        for i in range(epoch):
            print("\r\033[0Klearning in batch mode... {}%".format(int((((i + (epoch * count)) +1) / (epoch * 4))*100)), end="")
            gradient = np.dot(x.T, (hypothesis(x, thetas) - y)) / m
            thetas -= learning_rate * gradient
        theta.append((thetas, house))
        count += 1
    print(end="\n")
    return theta

def stochastic_gradient_descent(X,Y):
    learning_rate = 0.1
    x = np.hstack((np.ones((X.shape[0],1)), X))
    m = x.shape[0]
    theta = []

    epoch = 0
    for house in np.unique(Y):
        thetas = np.zeros(x.shape[1])
        y = np.where(Y == house, 1, 0)
        for it in range(m):
            print("\it\033[0Klearning in stochastic mode... {}%".format(int((((it + (m * epoch)) +1) / (m * 4))*100)), end="")
            gradient = x[it] * (hypothesis(x[it], thetas.T) - y[it])
            thetas -= learning_rate * gradient
        theta.append((thetas, house))
        epoch += 1
    print(end="\n")
    return theta

def minibatch_gradient_descent(X,Y):
    learning_rate = 0.1
    x = np.hstack((np.ones((X.shape[0],1)), X))
    m = x.shape[0]
    b = 5
    theta = []
    
    epoch = 0
    for house in np.unique(Y):
        thetas = np.zeros(x.shape[1])
        y = np.where(Y == house, 1, 0)
        for it in range(0, m, b):
            print("\r\033[0Klearning in mini-batch mode... {}%".format(int((((it+b + (m * epoch)) +1) / (m * 4))*100)), end="")
            gradient = np.dot(x[it:it+b].T, (hypothesis(x[it:it+b], thetas) - y[it:it+b]))
            thetas -= (learning_rate*1/b) * gradient
        theta.append((thetas, house))
        epoch += 1
    print(end="\n")
    return theta

def predict(X, thetas, houses):
    x = np.hstack((np.ones((X.shape[0],1)), X))
    Y_hat = np.array([[sigmoid(i.dot(theta)) for theta in thetas] for i in x])
    Y_hat = np.array([houses[np.argmax(j)] for j in Y_hat])
    return Y_hat

def accuracy(y_hat, x):
    acc = np.where(y_hat == x, 1, 0)
    print("Training is completed with an accuracy of {}%".format(round(acc.sum() / len(y_hat), 4) * 100))

"""
Main
"""

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print('Please run the program with \'--help\' as an argument for more information')
        exit(1)
    if sys.argv[1] == "--help":
        print('''
        [1stARG=Program Name][2ndARG=File Containing Data] [3rdARG(optional)=Traing Mode]\n
        \t2ndARG must be a .csv file.\n
        \t3rdARG may be:\n
        \t\t1: --minibatch for Mini-Batch Mode\n
        \t\t2: --stochastic for Stochastic Mode\n
        \t\t3: --batch for Batch Mode\n
        \t\t4: if no 3rd argument is passed the Batch Mode will be used as default
        ''')
        exit(0)
    try:
            data = pd.read_csv(sys.argv[1])
    except:
        print('File not found or it is corrupted.\nPlease, make sure that ./dataset_train.csv is available.')
        exit(2)

    try:
        X = data[[
            "Hogwarts House",
            "Defense Against the Dark Arts", 
            "Herbology", 
            "Ancient Runes",
            "Charms",
            "Flying",
            "Muggle Studies",
            "History of Magic",
            "Divination",
            "Astronomy",
            "Transfiguration"
        ]].dropna()

        X = X.sample(frac=1, random_state=1)

        Y = X["Hogwarts House"].to_numpy()

        X.drop(["Hogwarts House"], axis=1, inplace=True)
        X_norm = normalization(X, X.min(), X.max())

        X_train, X_test, Y_train, Y_test = train_test_split(X_norm,Y, test_size=0.2, random_state=1)

        if(len(sys.argv) >= 3):
            if(sys.argv[2] == "--minibatch"):
                theta = minibatch_gradient_descent(X_train,Y_train)
            elif(sys.argv[2] == "--stochastic"):
                theta = stochastic_gradient_descent(X_train,Y_train)
            elif(sys.argv[2] == "--batch"):
                theta = gradient_descent(X_train,Y_train)
            else:
                print("Mode was not recognized, training shall run in default(batch) mode")    
                theta = gradient_descent(X_train,Y_train)
        else:    
            theta = gradient_descent(X_train,Y_train)

        theta_dict = {column: row for row, column in theta}
        df_theta = pd.DataFrame(theta_dict)
        df_theta.to_csv("thetas.csv", index=False)

        thetas = df_theta.to_numpy()
        houses = df_theta.columns.to_list()
        Y_hat = predict(X_test, thetas.T, houses)
        accuracy(Y_hat, Y_test)
        print("Weights were exported to \'thetas.csv\' file")

    except Exception as e:
        print('Please, make sure that {} is well formatted'.format(sys.argv[1]))
        # print(e)
        exit(3)

