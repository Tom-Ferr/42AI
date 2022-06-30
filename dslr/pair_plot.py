import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.plotting import scatter_matrix



data = pd.read_csv('dataset_train.csv')
feats = data.dropna()
col = pd.DataFrame(feats.columns[6:])

scatter_matrix(data, alpha = 0.2, figsize = (13, 13), diagonal = 'kde')
# n = feats.shape[0]
# houses = ('Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff')
# colors = {'Ravenclaw': 'blue', 'Slytherin': 'green', 'Gryffindor': 'red', 'Hufflepuff': 'yellow'}

# fig, ax = plt.subplots(13,13)
# groups = feats.groupby('Hogwarts House')
# x = y = 0
# for j in range(13):
#     for i in houses:
#         ax[x][y].scatter(groups.get_group(i).loc[:,'Index'], groups.get_group(i).loc[:][col[j]], color=colors[i], alpha=0.3)
#     ax[x][y].title.set_text(col[j])
#     y += 1
#     if y >= 13:
#         y = 0
#         x += 1
# fig.delaxes(ax[3,1])
# fig.delaxes(ax[3,2])
# fig.delaxes(ax[3,3])
# fig.legend(labels=houses, bbox_to_anchor=(0.55, 0.2))
# fig.set_figwidth(15)
# fig.set_figheight(7)
# fig.tight_layout()
plt.show()
