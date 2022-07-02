import matplotlib.pyplot as plt
import pandas as pd


try:
    data = pd.read_csv('dataset_train.csv')
except:
    print('File not found or it is corrupted.\nPlease, make sure that ./dataset_train.csv is available.')
    exit(1)
try:
    feats = data.dropna()
    col = feats.columns[6:]
    n = col.shape[0]
    houses = ('Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff')
    colors = {houses[0]: 'blue', houses[1]: 'green', houses[2]: 'red', houses[3]: 'yellow'}

    fig, ax = plt.subplots(4,4)
    groups = feats.groupby('Hogwarts House')
    x = y = 0
    prog = '             ' #blank spaces
    print('\r\033[0KGathering data [{}]'.format(prog), end='')
    for j in range(n):
        for i in houses:
            ax[x][y].scatter(groups.get_group(i).loc[:,'Index'], groups.get_group(i).loc[:][col[j]], color=colors[i], alpha=0.3)
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
    fig.legend(labels=houses, bbox_to_anchor=(0.95, 0.2))
    fig.set_figwidth(15)
    fig.set_figheight(7)
    fig.tight_layout()
    plt.show()
except:
    print('Please, make sure that dataset_train.csv is well formatted')
    exit(2)
