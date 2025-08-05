#初始化教材的数据集以及环境
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
# Common imports
import numpy as np
import numpy.random as rnd
import os
# to make this notebook's output stable across runs
rnd.seed(42)
# To plot pretty figures
#matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
#import mnist.mat
from scipy.io import loadmat
mnist_raw = loadmat("mnist-original")
#print(mnist_raw.keys())
mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
print("mnist is imported.")
#print(mnist)
#initiate the data
X, y = mnist["data"], mnist["target"]
#print("the shape of data is:")
#print(X.shape)

#configure the matplotlib
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(9,9))

#initiate the training set and the testing set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = rnd.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#开始测试
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import confusion_matrix
#GridSearchCV
from sklearn.model_selection import GridSearchCV
knc = KNeighborsClassifier()
parameters = {'n_neighbors': [2, 4, 6],
              'weights': ['uniform', 'distance'],
              'n_jobs': [-1]
              }
grid_knc = GridSearchCV(knc, parameters, return_train_score=True, verbose=2)
print("grid_knc loaded")
#fit
grid_knc.fit(X_train, y_train)
print("grid_knc has fitted the data.")
print("best estimator of GridSearch: ")
print(grid_knc.best_estimator_)
#predict
y_test_predict = grid_knc.predict(X_test)
#evaluate
test_matrix = confusion_matrix(y_test, y_test_predict)
print("For testing set, data from test_matrix:")
for i in range(10):
    print(str(i) + ": " + str(test_matrix[i][i]))
#evaluate(with the number of pictures considered)
test_sum_row = test_matrix.sum(axis=1, keepdims=True)
test_aver_matrix = test_matrix / test_sum_row
print("For testing set, data from test_matrix: with the number of pictures considered:")
for i in range(10):
    print(str(i) + ": " + str(test_aver_matrix[i][i]))

#precision_score
from sklearn.metrics import precision_score
p_score = precision_score(y_test, y_test_predict, average='macro')
print("precision_score: ")
print(p_score)