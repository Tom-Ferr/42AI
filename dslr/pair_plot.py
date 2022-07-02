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

    fig, ax = plt.subplots(n,n)
    groups = feats.groupby('Hogwarts House')
    prog = '             ' #blank spaces

    print('\r\033[0KGathering data [{}]'.format(prog), end='')

    for y in range(n):
        for x in range(n):
            if x == y:
                for k in houses:
                    ax[x][y].hist(groups.get_group(k).loc[:][col[x]], bins=15, edgecolor='black',color=colors[k], alpha=0.3)
            else:           
                for k in houses:
                    ax[x][y].scatter(groups.get_group(k).loc[:][col[x]], groups.get_group(k).loc[:][col[y]], color=colors[k], alpha=0.1)
            if y == 0:
                ax[x][y].set_title(col[x], fontsize=6, x=-1, y=0.2)
                ax[x][y].yaxis.set_ticks_position('left')
            else:
                ax[x][y].yaxis.set_visible(False)
            if x == (n - 1):
                ax[x][y].set_xlabel(col[y], fontsize=5)
                ax[x][y].xaxis.set_ticks_position('bottom')
            else:
                ax[x][y].xaxis.set_visible(False)
        prog = prog.replace(" ", "-", 1)
        print('\r\033[0KGathering data [{}]'.format(prog), end='')
    print('\nPlotting...')
    fig.legend(labels=houses, bbox_to_anchor=(0.11, 0.15))
    fig.set_figwidth(15)
    fig.set_figheight(7.5)
    fig.subplots_adjust(left=0.13,
                        bottom=0.15, 
                        right=0.99, 
                        top=0.99, 
                        wspace=0, 
                        hspace=0)
    plt.show()
except:
    print('Please, make sure that dataset_train.csv is well formatted')
    exit(2)
