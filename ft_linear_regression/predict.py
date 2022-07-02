import matplotlib.pyplot as plt
import pandas as pd
import sys
try:
    thetas = pd.read_csv('.thetas.csv')
except:
    print('Pelase, run train.py first')
    exit(1)
try:
    t0 = thetas.loc[0][0]
    t1 = thetas.loc[0][1]

    X_hat = int(input("Please, enter your car's mileage, then I'll predict a price for it:\n"))
    Y_hat = int(X_hat * t1 + t0)
    print("The estimate price of your car is {}".format(Y_hat))

    if len(sys.argv) > 1:
        if sys.argv[1] == "-b":
            data = pd.read_csv('data.csv')
            X = data.loc[0:]['km']
            Y = data.loc[0:]['price']
            hypothesis = X * t1 + t0

            plt.title("ft_linear_regression")
            plt.xlabel("km")
            plt.ylabel("price")
            plt.scatter(X,Y)
            plt.scatter(X_hat,Y_hat, color="red")
            plt.plot(X,hypothesis, color="black")
            plt.show()
except:
    print('Please, make sure that you ran train.py before this one')
    exit(2)