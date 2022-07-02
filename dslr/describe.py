import pandas as pd
import sys
import math
import numpy as np

def linear_interpolation(value, mini, maxi):
    return (maxi - mini) * value + mini

def ft_percentile(df, x):
    n = len(df)
    p = x/100 * (n + 1)
    df = np.array(df)
    df.sort()
    i = int(p)
    p = p - i
    j = i - 1
    return linear_interpolation(p, df[i], df[j])


def create_desc():
    i_ls = data.columns
    desc = {}
    for i in i_ls:
        desc[i] = []
    return i_ls, desc

def description_gen():
    i_ls, desc = create_desc()
    for j in range(m):
        try:
            column = np.array(data.iloc[:,j])
            count = summ = summ_sq = 0.0
            mini = maxi = column[0]
            trash = []
            for i in range(n):
                if pd.isnull(column[i]):
                    trash.append(i)
                    continue
                count += 1
                summ_sq += column[i]**2
                summ += column[i]
                if column[i] < mini:
                    mini = column[i]
                if column[i] > maxi:
                    maxi = column[i]
            column = np.delete(column, trash)
            mean = summ / count
            std = summ_sq - (summ**2 / count)
            std = math.sqrt(std / (count - 1))
            desc[i_ls[j]].append(count)
            desc[i_ls[j]].append(mean)
            desc[i_ls[j]].append(std)
            desc[i_ls[j]].append(mini)
            for x in range(25, 100, 25):
                desc[i_ls[j]].append(ft_percentile(column, x))
            desc[i_ls[j]].append(maxi)
            #bonus
            desc[i_ls[j]].append(maxi - mini)   #range
            iqr = ft_percentile(column, 75) - ft_percentile(column, 25)
            desc[i_ls[j]].append(iqr)
            vari = summ_sq - (summ**2 / (count - 1))
            vari = math.sqrt(std / count)
            desc[i_ls[j]].append(vari)
        except:
            desc.pop(i_ls[j])
            continue
    return desc

"""
Main
"""

if len(sys.argv) > 1:
    try:
        data = pd.read_csv(sys.argv[1])
    except:
        print('File was not found or it is corrupted')
        exit(1)
    n = data.shape[0]
    m = data.shape[1]
    desc = description_gen()
    desc =  pd.DataFrame(desc, index = ['Count','Mean','Std','Min','25%','50%','75%','Max',
    'Range', 'IQR', 'Variance'])
    print(desc)
else:
    print('No, dataset to describe')

    