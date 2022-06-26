import pandas as pd
import sys
import math

def percentile(x):
    p = x * (n + 1) / 100
    return round(p)

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
            count = summ = summ_sq = 0
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
            column.sort_values()
            for x in range(25, 100, 25):
                desc[i_ls[j]].append(column[percentile(x)])
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
    