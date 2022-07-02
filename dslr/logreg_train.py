import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print('Please pass a .csv file as a parameter')
    exit()
try:
    data = pd.read_csv(sys.argv[1])
except:
    print('File not found or it is corrupted.\nPlease, make sure that ./dataset_train.csv is available.')
    exit()

