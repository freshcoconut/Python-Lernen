import random, os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#设置随机生成器的种子
np.random.seed(30)
random.seed(30)

mpl.rc('axes', labelsize=16)
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)

#try to make directory
def try_mkdir(dir_path):
    if not os.path.exists(dir_path):
        print("Making directory(s): " + dir_path + ".")
        os.makedirs(dir_path)
    
#save the picture
def save_fig(name, path, tight=True, exten="png", resolution=300):
    path_and_file = os.path.join(path, name + "." + exten)
    try_mkdir(path)#prepare the directory
    print("Saving", name  + ".")
    if tight:
        plt.tight_layout()
    plt.savefig(path_and_file, format=exten, dpi=resolution)

#获得纯数字的Series
def chooseNum(series, cat):#cat: cat_attribute
    resultNum = series.drop(cat, axis=1)
    return resultNum

#数组一删去与数组二相同的元素
def minus(arr1, arr2):
    return [x for x in arr1 if x not in arr2]

#遍历一个一维数组，统计出现了哪几种值。用数字给这些值编号（索引值）
#为了使数组维持pandas的结构，程序并不调用sklearn中已有的方法。因为那些方法会将pandas数据文件转化为numpy数据文件，这样会丢失很多信息。
#相当于onehot，给出现的值编号
'''
例如，
arr1 = ['tom', 'dad', 'cat', 'dad', 'dad']，
那么我们可以得到两个数组 x 和 name ：
name = ['tom', 'dad', 'cat'] （相当于 tom 对应 0， ’dad‘ 对应 1， 'cat' 对应 2）
x = [ 0, 1, 2, 1, 1]
'''
def assign_name(arr):
    name = []
    x = []
    for e in arr:
        if e in name:
            e_index = name.index(e)
            x.append(e_index)
        else:
            name.append(e)
            x.append( len(name) - 1 )
    return name, x

#批量输出图片
#num
def num_visual_plot(data_set):#data_set特指train_data_num
    num_path_png = "./house_description/png/num_plot/"
    num_path_eps = "./house_description/eps/num_plot/"
    column_name = data_set.columns.values
    for cname in column_name[:-1]:       
        data_set.plot(kind="scatter", x=cname, y="SalePrice")
        save_fig("num_png_" + cname + "_visual_plot", num_path_png, exten="png")#png
        save_fig("num_eps_" + cname + "_visual_plot", num_path_eps, exten="eps")#eps
        plt.cla()#清除之前的绘图
        plt.close()#防止出现RuntimeWarning: More than 20 figures have been opened.

#cat
def cat_visual_scatter(data_set):#data_set特指train_data_cat
    cat_path_png = "./house_description/png/cat_scatter/"
    cat_path_eps = "./house_description/eps/cat_scatter/"
    column_name = data_set.columns.values
    #建立一个用于保存column_name的csv文件
    name_list = []
    #批量绘图
    for cname in column_name[:-1]:
        plt.xlabel(cname)
        plt.ylabel("SalePrice")
        x_name, x = assign_name(data_set[cname])
        #储存x_name
        name_list.append(x_name)
        #画图
        y_data_price = data_set["SalePrice"]
        plt.scatter(x, y_data_price)
        save_fig("cat_png_" + cname + "_visual_scatter", cat_path_png, exten="png")#png
        save_fig("cat_eps_" + cname + "_visual_scatter", cat_path_eps, exten="eps")#eps
        plt.cla()#清除之前的绘图
        plt.close()#防止出现RuntimeWarning: More than 20 figures have been opened.
    #保存name_list --> csv
    name_pre_csv = pd.DataFrame({'subject': column_name[:-1], 'name': name_list})
    print("saving " + "索引所对应的标签.csv")
    name_pre_csv.to_csv(cat_path_png + "索引所对应的标签.csv", index=False, sep=',')
    name_pre_csv.to_csv(cat_path_eps + "索引所对应的标签.csv", index=False, sep=',')
        
#Now read the data from *.csv
data_1 = "train.csv"
data_2 = "test.csv"
data_3 = "sample_submission.csv"

#cat_attribute
cat_att = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]
#影响较小的因素不予考虑，不然r2_score会变成负数，随机森林能跑出-17的“好成绩”
#cat_ignore = ["Street", "Alley", "LandContour", "Utilities", "LandSlope", "Condition1", "Condition2", "RoofMatl", "Heating", "Functional", "PoolQC", "MiscFeature", "LotShape", "BsmtFinType1", "BsmtFinType2"]
cat_ignore = []

#training set
#number
train_data = (pd.read_csv(data_1)).drop("Id", axis=1)
train_data_num = chooseNum(train_data, cat_att)

#统计数字数据中各个数值的出现次数
hist_path_png = "./house_description/png/"
hist_path_eps = "./house_description/eps/"
train_data_num.hist(bins=50, figsize=(20,15))
save_fig("data_num_counts", path=hist_path_png, exten="png")#png
save_fig("data_num_counts", path=hist_path_eps, exten="eps")#eps
plt.cla()#清除之前的绘图

#cat
cat_and_price = minus(cat_att, cat_ignore) + ["SalePrice"]
train_data_cat = train_data[cat_and_price]

#批量输出num和cat的plot图像
num_visual_plot(train_data_num)
cat_visual_scatter(train_data_cat)

