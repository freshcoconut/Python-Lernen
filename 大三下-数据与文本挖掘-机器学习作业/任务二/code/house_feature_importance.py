import random, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

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
        
#用随机森林判断各个变量的重要程度
from sklearn.ensemble import RandomForestRegressor

#设置随机生成器的种子
np.random.seed(30)
random.seed(30)

#define the methods
#获得纯数字的Series
def chooseNum(series, cat):#cat: cat_attribute
    resultNum = series.drop(cat, axis=1)
    return resultNum

#去除不需要的数字变量
def removeNum(series, num):#num: num_ignore
    resultNum = series.drop(num, axis=1)
    return resultNum

#数组一删去与数组二相同的元素
def minus(arr1, arr2):
    return [x for x in arr1 if x not in arr2]

#将训练集和测试集拼起来，进行OneHotEncode，然后分开
def onehot_preprocessing(train_set, test_set, cat, cat_ig=[]):#cat: cat_attribute; cat_ignore=[] in default settings
    #pointw: point_where_test_set_starts
    pointw = (train_set.shape)[0]
    whole_set = train_set.append(test_set, ignore_index=True)
    #onehotencoder
    oh_coder = OneHotEncoder(sparse=False)
    cat_in_use = minus(cat, cat_ig)
    whole_set_onehot = oh_coder.fit_transform(whole_set[cat_in_use])
    train_set_onehot = whole_set_onehot[:pointw]
    test_set_onehot = whole_set_onehot[pointw:]
    return train_set_onehot, test_set_onehot

#获得训练集和测试集中纯数字的数据，并用Pipeline完成标准化操作
def num_preprocessing(train_set, test_set, cat, num_ig=[]):#cat: cat_attribute; num_ignore=[] in default settings
    #remove cat_attributes
    train_set_num_pre_pre = chooseNum(train_set, cat)
    test_set_num_pre_pre = chooseNum(test_set, cat)
    #remove digital columns that we do not need
    train_set_num_pre = removeNum(train_set_num_pre_pre, num_ig)
    test_set_num_pre = removeNum(test_set_num_pre_pre, num_ig)    
    #Pipeline
    pipe_num = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])
    #fit_transform
    train_set_num = pipe_num.fit_transform(train_set_num_pre)
    test_set_num = pipe_num.fit_transform(test_set_num_pre)
    return train_set_num, test_set_num

#进一步简化：一步完成数字与文本的标准化
def formal_transform(train_set, test_set, cat, cat_ig=[], num_ig=[]):#cat: cat_attribute; cat_ignore=[] and num_ignore=[] in default settings
    train_num, test_num = num_preprocessing(train_set, test_set, cat, num_ig)
    train_cat, test_cat = onehot_preprocessing(train_set, test_set, cat, cat_ig)
    print("train_num_shape: " + str(train_num.shape))
    print("train_cat_shape: " + str(train_cat.shape))
    train_prepared = np.concatenate((train_num, train_cat), axis=1)
    test_prepared = np.concatenate((test_num, test_cat), axis=1)
    return train_prepared, test_prepared

#Now read the data from *.csv
data_1 = "train.csv"
data_2 = "test.csv"
data_3 = "sample_submission.csv"

#training set
train_data = pd.read_csv(data_1)
X_train = train_data.drop("SalePrice", axis=1)
y_train = (train_data.loc[:, ["SalePrice"]]).to_numpy().ravel()#ravel()用于降维

#testing set
X_test = pd.read_csv(data_2)
y_test = (pd.read_csv(data_3).loc[:, ["SalePrice"]]).to_numpy().ravel()#ravel()用于降维

#cat_attribute
cat_att = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]

#影响较小的因素不予考虑，不然r2_score会变成负数，随机森林能跑出-17的“好成绩”
cat_ignore = ["Utilities", "PoolQC"]
num_ignore = ["ScreenPorch", "PoolArea", "MiscVal", "YrSold", "BsmtUnfSF", "2ndFlrSF", "LotArea", "LotFrontage", "MasVnrArea", "WoodDeckSF"]

#prepare the sets
print("Preparing and refining the sets.")

X_train_prepared, X_test_prepared = formal_transform(X_train, X_test, cat_att, cat_ignore, num_ignore)

print("Sets are prepared and refined.")

#随机森林进行拟合
rfr = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=30)
rfr.fit(X_train_prepared, y_train)

feature_name = X_train.columns.values
fi_zip = pd.DataFrame(np.array(rfr.feature_importances_), columns=['feature importance'])

#保存fi_zip --> csv
fi_path = "./house_feature_importance/"
try_mkdir(fi_path)
print("Saving feature.csv.")
fi_zip.to_csv(fi_path + "feature.csv", index=True, sep=',')
#sorting feature_importance
fi_zip_read = pd.read_csv(fi_path + "feature.csv")
fi_zip_sorted = fi_zip_read.sort_values(by="feature importance", ascending=False)
print("Saving feature_sorted.csv.")
fi_zip_sorted.to_csv(fi_path + "feature_sorted.csv", index=False, sep=',')
#给特征值的重要程度绘图
plt.cla()#清除之前可能存在的绘图
x_value = fi_zip_sorted["Unnamed: 0"]
y_value = fi_zip_sorted["feature importance"]
plt.title("Distribution of Feature Importance")
plt.xlabel("Index")
plt.ylabel("Feature Importance")
plt.scatter(x_value, y_value)
save_fig("scatter_feature_importance", fi_path)#png
plt.cla()#清除之前的绘图
