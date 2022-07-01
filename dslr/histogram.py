import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('dataset_train.csv')

feats = data.dropna()
n = feats.shape[0]
bins = 15
col = feats.columns[6:]
houses = ('Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff')
colors = {'Ravenclaw': 'blue', 'Slytherin': 'green', 'Gryffindor': 'red', 'Hufflepuff': 'yellow'}

fig, ax = plt.subplots(4,4)
groups = feats.groupby('Hogwarts House')
x = y = 0
prog = '             ' #blank spaces
print('\r\033[0KGathering data [{}]'.format(prog), end='')
for j in range(13):
    for i in houses:
        ax[x][y].hist(groups.get_group(i).loc[:][col[j]], bins=bins, edgecolor='black',color=colors[i], alpha=0.3)
    ax[x][y].title.set_text(col[j])
    y += 1
    if y >= 4:
        y = 0
        x += 1
    prog = prog.replace(" ", "-", 1)
    print('\r\033[0KGathering data [{}]'.format(prog), end='')
print('\nPlotting...')
fig.delaxes(ax[3,1])
fig.delaxes(ax[3,2])
fig.delaxes(ax[3,3])
fig.legend(labels=houses, bbox_to_anchor=(0.55, 0.2))
fig.set_figwidth(15)
fig.set_figheight(7)
fig.tight_layout()
plt.show()