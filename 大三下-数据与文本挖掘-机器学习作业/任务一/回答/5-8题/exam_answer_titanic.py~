#附着在运行环境（模板）后，在命令行条件下运行
#！！！该文件只用于记录代码以及运行结果，无法直接运行！！！
#关于实际运行的程序，需参见题目相应的文件

#question 3: tackle the Titanic database
import numpy as np
import pandas as pd

#initiate the database
#train_all
train_all = pd.read_csv('train.csv')
y_train = train_all['Survived']
#test_all
test_all = pd.read_csv('test.csv')
y_test_all = pd.read_csv('gender_submission.csv')
y_test = y_test_all['Survived']# y for test in use
#pclass
X_train_pclass = train_all['Pclass']
X_test_pclass = test_all['Pclass']
#sex
temp_train_sex = train_all['Sex']
X_train_sex = pd.array([])
for i in range(temp_train_sex.size):
    if temp_train_sex[i] == "male":
        X_train_sex = np.append(X_train_sex, 1)
    else:
        X_train_sex = np.append(X_train_sex, 0)

temp_test_sex = test_all['Sex']
X_test_sex = pd.array([])
for i in range(temp_test_sex.size):
    if temp_test_sex[i] == "male":
        X_test_sex = np.append(X_test_sex, 1)
    else:
        X_test_sex = np.append(X_test_sex, 0)

#age
#用中位数填补缺失的年龄
median_age_train = train_all['Age'].median()
X_train_age = train_all['Age']
X_train_age.fillna(median_age_train, inplace=True)

median_age_test = test_all['Age'].median()
X_test_age = test_all['Age']
X_test_age.fillna(median_age_test, inplace=True)
#sibsp
X_train_sibsp = train_all['SibSp']
X_test_sibsp = test_all['SibSp']
#parch
X_train_parch = train_all['Parch']
X_test_parch = test_all['Parch']
#fare
median_fare_train = train_all['Fare'].median()
X_train_fare = train_all['Fare']
X_train_fare.fillna(median_fare_train, inplace=True)

median_fare_test = test_all['Fare'].median()
X_test_fare = test_all['Fare']
X_test_fare.fillna(median_fare_test, inplace=True)
#embarked
temp_train_embarked = train_all['Embarked']
X_train_embarked = pd.array([])
for i in range(temp_train_embarked.size):
    if temp_train_embarked[i] == "Q":
        X_train_embarked = np.append(X_train_embarked, 0)
    elif temp_train_embarked[i] == "S":
        X_train_embarked = np.append(X_train_embarked, 1)
    else:
        X_train_embarked = np.append(X_train_embarked, 2)

temp_test_embarked = test_all['Embarked']
X_test_embarked = pd.array([])
for i in range(temp_test_embarked.size):
    if temp_test_embarked[i] == "Q":
        X_test_embarked = np.append(X_test_embarked, 0)
    elif temp_test_embarked[i] == "S":
        X_test_embarked = np.append(X_test_embarked, 1)
    else:
        X_test_embarked = np.append(X_test_embarked, 2)

#expend dimension
#升维，例如：[a, b] -> [[a], [b]]
#pclass + sex + age + sibsp + parch + fare + embarked

#numpy以后会把这个功能(multi-dimensional)删掉，不过不影响我现在使用它
#FutureWarning: Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.
#上面这个警告无伤大雅

X_train_pclass_ed = X_train_pclass[:, None]
X_test_pclass_ed = X_test_pclass[:, None]

X_train_sex_ed = X_train_sex[:, None]
X_test_sex_ed = X_test_sex[:, None]

X_train_age_ed = X_train_age[:, None]
X_test_age_ed = X_test_age[:, None]

X_train_sibsp_ed = X_train_sibsp[:, None]
X_test_sibsp_ed = X_test_sibsp[:, None]

X_train_parch_ed = X_train_parch[:, None]
X_test_parch_ed = X_test_parch[:, None]

X_train_fare_ed = X_train_fare[:, None]
X_test_fare_ed = X_test_fare[:, None]

X_train_embarked_ed = X_train_embarked[:, None]
X_test_embarked_ed = X_test_embarked[:, None]

#pclass + sex + age + sibsp + parch + fare + embarked
#X_train
X_train = np.hstack((X_train_pclass_ed, X_train_sex_ed))
X_train = np.hstack((X_train, X_train_age_ed))
X_train = np.hstack((X_train, X_train_sibsp_ed))
X_train = np.hstack((X_train, X_train_parch_ed))
X_train = np.hstack((X_train, X_train_fare_ed))
X_train = np.hstack((X_train, X_train_embarked_ed))
print("X_train.shape:")
print(X_train.shape)
#X_test
X_test = np.hstack((X_test_pclass_ed, X_test_sex_ed))
X_test = np.hstack((X_test, X_test_age_ed))
X_test = np.hstack((X_test, X_test_sibsp_ed))
X_test = np.hstack((X_test, X_test_parch_ed))
X_test = np.hstack((X_test, X_test_fare_ed))
X_test = np.hstack((X_test, X_test_embarked_ed))
print("X_test.shape:")
print(X_test.shape)

#y shape
print("y_train.shape:")
print(y_train.shape)
print("y_test.shape:")
print(y_test.shape)


#train the models
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
lr =  LinearRegression()
parameters ={
    "fit_intercept": [True, False]
}
grid_lr = GridSearchCV(lr, parameters, n_jobs=-1, verbose=1)
#train
print("start fitting")
grid_lr.fit(X_train, y_train)
print("fitted")

#pclass predict
print("grid_lr start predicting")
y_predict = grid_lr.predict(X_test)
print("grid_lr has predicted")

#evaluate
#print("The probability predicted to survive:")
#print(y_predict)

#score
score = grid_lr.score(X_test, y_test)
print()
print("grid_lr.score(X_test, Y_test):")
print(score)

#程序末端
print()
print("numpy以后会把这个功能(multi-dimensional)删掉，不过不影响我现在使用它")
print("（FutureWarning）这个警告无伤大雅")

#result
'''
X_train.shape:
(891, 7)
X_test.shape:
(418, 7)
y_train.shape:
(891,)
y_test.shape:
(418,)
titanic.py:84: FutureWarning: Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.
  X_train_pclass_ed = X_train_pclass[:, None]
titanic.py:85: FutureWarning: Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.
  X_test_pclass_ed = X_test_pclass[:, None]
titanic.py:90: FutureWarning: Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.
  X_train_age_ed = X_train_age[:, None]
titanic.py:91: FutureWarning: Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.
  X_test_age_ed = X_test_age[:, None]
titanic.py:93: FutureWarning: Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.
  X_train_sibsp_ed = X_train_sibsp[:, None]
titanic.py:94: FutureWarning: Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.
  X_test_sibsp_ed = X_test_sibsp[:, None]
titanic.py:96: FutureWarning: Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.
  X_train_parch_ed = X_train_parch[:, None]
titanic.py:97: FutureWarning: Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.
  X_test_parch_ed = X_test_parch[:, None]
titanic.py:99: FutureWarning: Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.
  X_train_fare_ed = X_train_fare[:, None]
titanic.py:100: FutureWarning: Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.
  X_test_fare_ed = X_test_fare[:, None]
start fitting
Fitting 5 folds for each of 2 candidates, totalling 10 fits
fitted
grid_lr start predicting
grid_lr has predicted

grid_lr.score(X_test, Y_test):
0.6895639772220317

numpy以后会把这个功能(multi-dimensional)删掉，不过不影响我现在使用它
（FutureWarning）这个警告无伤大雅

Process finished with exit code 0
'''
