#初始化教材的数据集以及环境
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import os

# Common imports
import numpy as np
import numpy.random as rnd

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
print("mnist is imported")

#initiate the data
X, y = mnist["data"], mnist["target"]

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
from PIL import Image
#mnist数据集内，图片是(1, 784)格式，需要先将其转化为(28, 28)格式——这是教材的处理方式

def change_pic(pic_file, orientation, cut_num):#输入nparray object的unit8图片， 输出nparray object
    #mnist数据集内，图片是(1, 784)格式，需要先将其转化为(28, 28)格式——这是教材的处理方式
    #28可以替换为对图片长度求平方根，这里为了省事，就不改了
    pic_file_reshape = pic_file.reshape((28,28))
    pic = Image.fromarray(pic_file_reshape)#pic_file(ndarray object) is transformed to pic(PIL.Image object)
    xsize, ysize = pic.size
    # orientation: 1-up; 2-down; 3-left; 4-right
    #移动cut_num个像素
    if orientation==1:#up
        (left, upper, right, lower) = (0, cut_num, xsize, ysize)
        pic_crop = pic.crop((left, upper, right, lower))#transform
        prepre_result = np.asarray(pic_crop).reshape(( (28 - cut_num) , 28))#矩形
        pre_result = np.vstack((prepre_result, np.zeros((cut_num, 28))))#正方形
    elif orientation==2:#down
        (left, upper, right, lower) = (0, 0, xsize, ysize - cut_num)
        pic_crop = pic.crop((left, upper, right, lower))#transform
        prepre_result = np.asarray(pic_crop).reshape(((28 - cut_num), 28))#矩形
        pre_result = np.vstack((np.zeros((cut_num, 28)), prepre_result))  # 正方形
    elif orientation==3:#left
        (left, upper, right, lower) = (cut_num, 0, xsize, ysize)
        pic_crop = pic.crop((left, upper, right, lower))#transform
        prepre_result = np.asarray(pic_crop).reshape((28, (28 - cut_num)))#矩形
        pre_result = np.hstack((prepre_result, np.zeros((28, cut_num))))#正方形
    elif orientation==4:#right
        (left, upper, right, lower) = (0, 0, xsize - cut_num, ysize)
        pic_crop = pic.crop((left, upper, right, lower))#transform
        prepre_result = np.asarray(pic_crop).reshape((28, (28 - cut_num)))  # 矩形
        pre_result = np.hstack((np.zeros((28, cut_num)), prepre_result))#正方形
    #return
    result = np.asarray(pre_result).reshape((28 * 28))  #还原为跟之前一样的数组
    return result

#enhance the training set
print("start to enhancing the training sets")
X_train_strong = X_train
y_train_strong = y_train

#在四个方向上各移动[3, 6]个像素
for i in range(1,5):#四个方向
    for cut in [3, 6]:#cut_num
        #enhance X_train set
        X_train_temp = np.array([change_pic(x, i, cut) for x in X_train])
        X_train_strong = np.concatenate((X_train_strong, X_train_temp))
        #enhance y_train set
        y_train_strong = np.concatenate((y_train_strong, y_train))
        print("在方向" + str(i) + "完成" + str(cut) + "个像素的剪切")

print("The sets have been enhanced.")

#training the model
#借用第一题的代码
from sklearn.neighbors import  KNeighborsClassifier
#由第一题可知：什么是最好的模型
best_knc = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=-1)
print("best_knc loaded")
print("start fitting")
best_knc.fit(X_train_strong, y_train_strong)
print("fitting done")
#检验预测效果
print("start predicting")
y_predict = best_knc.predict(X_test)
print("predicting done")
#evaluate
from sklearn.metrics import confusion_matrix
test_matrix = confusion_matrix(y_test, y_predict)

test_sum_row = test_matrix.sum(axis=1, keepdims=True)
test_aver_matrix = test_matrix / test_sum_row
print("For testing set, data from test_matrix: with the number of pictures considered:")
for i in range(10):
    print(str(i) + ": " + str(test_aver_matrix[i][i]))

#借用第一题的代码
#precision_score
from sklearn.metrics import precision_score
p_score = precision_score(y_test, y_predict, average='macro')
print("precision_score: ")
print(p_score)
