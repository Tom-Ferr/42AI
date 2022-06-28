import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('dataset_train.csv')

feats = data.dropna()
n = feats.shape[0]
col = feats.columns[6:]


dis = {}
p = 0
print('\r\033[0KGathering data...(0%)', end='')
for j in col:
    dis[j] = {}
    houses = {'Ravenclaw': [], 'Slytherin': [], 'Gryffindor': [], 'Hufflepuff': []}
    for i in range(n):
        try:
            if  feats.loc[i]['Hogwarts House'] == 'Ravenclaw':
                houses['Ravenclaw'].append(feats.loc[i][j])
            elif feats.loc[i]['Hogwarts House'] == 'Slytherin':
                houses['Slytherin'].append(feats.loc[i][j])
            elif feats.loc[i]['Hogwarts House'] == 'Gryffindor':
                houses['Gryffindor'].append(feats.loc[i][j])
            elif feats.loc[i]['Hogwarts House'] == 'Hufflepuff':
                houses['Hufflepuff'].append(feats.loc[i][j])
        except:
            continue
    dis[j] = houses
    p += 1
    print('\r\033[0KGathering data...({}%)'.format(int(p/13 * 100)), end='')
fig, ax = plt.subplots(3,4)
print('\nPloting...')

m = 0
for j in range(4):
    for i in range(3):
        ax[i][j].hist(dis[col[m]]['Ravenclaw'], bins=15, color='blue', edgecolor='black', alpha=0.3)
        ax[i][j].hist(dis[col[m]]['Slytherin'], bins=15, color='green', edgecolor='black', alpha=0.3)
        ax[i][j].hist(dis[col[m]]['Gryffindor'], bins=15, color='red', edgecolor='black', alpha=0.3)
        ax[i][j].hist(dis[col[m]]['Hufflepuff'],  bins=15, color='yellow', edgecolor='black', alpha=0.3)
        m += 1
fig.legend(labels=('Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'))
plt.show()


