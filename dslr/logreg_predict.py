import numpy as np
import pandas as pd
from logreg_train import normalization, sigmoid, predict
import sys

def read_thetas():
    thetas = pd.read_csv(sys.argv[2])
    houses = thetas.columns.to_list()
    if thetas.shape[0] != 11 or thetas.shape[1] != 4:
        raise Exception 
    return thetas.to_numpy()

def get_normalized_data(data):
    X = data[[
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
        ]].fillna(0)

    return normalization(X, X.min(), X.max())

def export_predictions(Y_hat):
    out = [[i, Y_hat[i]] for i in range(len(Y_hat))]
    predictions = pd.DataFrame(out, columns=['Index','Hogwarts House'])
    predictions.to_csv("houses.csv", index=False)
    print("Ta da! Students houses shall be known as in \'houses.csv\' file")

"""
Main
"""

if __name__ == "__main__":

    """
    Pre-Checks
    """

    if len(sys.argv) < 2:
        print('Please run the program with \'--help\' as an argument for more information')
        exit(1)
    if sys.argv[1] == "--help":
        print('''
        [1stARG=Program Name][2ndARG=File Containing Data] [3rdARG=Exported File from \'logreg_train.py\']\n
        \t2ndARG must be a .csv file.\n
        \t3rdARG must be a .csv file generated by a previous run of \'logreg_train.py\'
        ''')
        exit(0)
    if len(sys.argv) < 3:
        print('Please pass a .csv file for the predictions AND the file exported by \'logreg_train.py\' as the second parameter')
        exit(2)

    """
    Read File
    """

    try:
        data = pd.read_csv(sys.argv[1])
    except:
        print('File not found or it is corrupted.\nPlease, make sure that ./dataset_test.csv is available.')
        exit(3)
    try:
        thetas = pd.read_csv(sys.argv[2])
        houses = thetas.columns.to_list()
        if thetas.shape[0] != 11 or thetas.shape[1] != 4:
            raise Exception 
        thetas = thetas.to_numpy()
    except Exception as e:
        print("Please, make sure to run \'logreg_train.py first\' AND that the file exported by it is well formatted, then pass it as a second parameter")
        exit(4)

    """
    Code
    """

    try:
        
        X_norm = get_normalized_data(data)

        print("Classifying students...")

        Y_hat = predict(X_norm,thetas.T, houses)

        export_predictions(Y_hat)
        

    except Exception as e:
        print('Please, make sure that {} is well formatted'.format(sys.argv[1]))
        # print(e)
        exit(5)