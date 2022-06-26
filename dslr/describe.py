import pandas as pd
import sys
import math
import numpy as np

def linear_interpolation(value, mini, maxi):
    return (maxi - mini) * value + mini

def normalization(value, mini, maxi):
    return (value - mini) / (maxi -  mini)

def percentile(df, x):
    p = x/100 * (n + 1)
    df = np.array(df)
    df.sort()
    if p.is_integer():
        return df[p]
    i = int(p)
    p = p - i
    j = i - 1
    return linear_interpolation(p, df[i], df[j])


def create_desc():
    i_ls = []
    for i in range(m):
        i_ls.append("Feature {}".format(i+1))
    desc = {}
    for i in i_ls:
        desc[i] = []
    return i_ls, desc

def description_gen():
    i_ls, desc = create_desc()
    for j in range(m):
        try:
            column = data.iloc[:,j]
            count = summ = summ_sq = 0.0
            mini = maxi = column[0]
            
            for i in range(n):
                if not pd.isnull(column[i]):
                    count += 1
                summ_sq += column[i]**2
                summ += column[i]
                if column[i] < mini:
                    mini = column[i]
                if column[i] > maxi:
                    maxi = column[i]
            mean = summ / n
            std = summ_sq - (summ**2 / n)
            std = math.sqrt(std / (n - 1))
            desc[i_ls[j]].append(count)
            desc[i_ls[j]].append(mean)
            desc[i_ls[j]].append(std)
            desc[i_ls[j]].append(mini)
            for x in range(25, 100, 25):
                desc[i_ls[j]].append(percentile(column, x))
            desc[i_ls[j]].append(maxi)
        except:
            desc.pop(i_ls[j])
            continue
    return desc

"""
Main
"""

if len(sys.argv) > 1:
    data = pd.read_csv(sys.argv[1])
    n = data.shape[0]
    m = data.shape[1]
    desc = description_gen()
    desc =  pd.DataFrame(desc, index = ['Count','Mean','Std','Min','25%','50%','75%','Max'])
    print(desc)
    print(data.describe())
    