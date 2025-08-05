#附着在运行环境（模板）后，在命令行条件下运行
#！！！该文件只用于记录代码以及运行结果，无法直接运行！！！
#关于实际运行的程序，需参见题目相应的文件
#question 1: KNeighborsClassifier
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

#result
#注：这个程序非常慢！！！非常非常慢！！！
'''
mnist is imported.
grid_knc loaded
Fitting 5 folds for each of 6 candidates, totalling 30 fits
[CV] END ..........n_jobs=-1, n_neighbors=2, weights=uniform; total time=  33.9s
[CV] END ..........n_jobs=-1, n_neighbors=2, weights=uniform; total time=  34.1s
[CV] END ..........n_jobs=-1, n_neighbors=2, weights=uniform; total time=  42.8s
[CV] END ..........n_jobs=-1, n_neighbors=2, weights=uniform; total time=  32.0s
[CV] END ..........n_jobs=-1, n_neighbors=2, weights=uniform; total time=  34.7s
[CV] END .........n_jobs=-1, n_neighbors=2, weights=distance; total time=  38.7s
[CV] END .........n_jobs=-1, n_neighbors=2, weights=distance; total time=  34.2s
[CV] END .........n_jobs=-1, n_neighbors=2, weights=distance; total time=  40.7s
[CV] END .........n_jobs=-1, n_neighbors=2, weights=distance; total time=  38.5s
[CV] END .........n_jobs=-1, n_neighbors=2, weights=distance; total time=  36.4s
[CV] END ..........n_jobs=-1, n_neighbors=4, weights=uniform; total time=  35.4s
[CV] END ..........n_jobs=-1, n_neighbors=4, weights=uniform; total time=  36.8s
[CV] END ..........n_jobs=-1, n_neighbors=4, weights=uniform; total time=  39.9s
[CV] END ..........n_jobs=-1, n_neighbors=4, weights=uniform; total time=  35.8s
[CV] END ..........n_jobs=-1, n_neighbors=4, weights=uniform; total time=  39.2s
[CV] END .........n_jobs=-1, n_neighbors=4, weights=distance; total time=  38.8s
[CV] END .........n_jobs=-1, n_neighbors=4, weights=distance; total time=  39.1s
[CV] END .........n_jobs=-1, n_neighbors=4, weights=distance; total time=  44.7s
[CV] END .........n_jobs=-1, n_neighbors=4, weights=distance; total time=  37.5s
[CV] END .........n_jobs=-1, n_neighbors=4, weights=distance; total time=  39.8s
[CV] END ..........n_jobs=-1, n_neighbors=6, weights=uniform; total time=  38.4s
[CV] END ..........n_jobs=-1, n_neighbors=6, weights=uniform; total time=  38.8s
[CV] END ..........n_jobs=-1, n_neighbors=6, weights=uniform; total time=  40.1s
[CV] END ..........n_jobs=-1, n_neighbors=6, weights=uniform; total time=  38.3s
[CV] END ..........n_jobs=-1, n_neighbors=6, weights=uniform; total time=  40.9s
[CV] END .........n_jobs=-1, n_neighbors=6, weights=distance; total time=  39.7s
[CV] END .........n_jobs=-1, n_neighbors=6, weights=distance; total time=  32.7s
[CV] END .........n_jobs=-1, n_neighbors=6, weights=distance; total time=  31.0s
[CV] END .........n_jobs=-1, n_neighbors=6, weights=distance; total time=  31.7s
[CV] END .........n_jobs=-1, n_neighbors=6, weights=distance; total time=  32.9s
grid_knc has fitted the data.
best estimator of GridSearch: 
KNeighborsClassifier(n_jobs=-1, n_neighbors=4, weights='distance')
For testing set, data from test_matrix:
0: 973
1: 1132
2: 995
3: 974
4: 950
5: 862
6: 946
7: 994
8: 920
9: 968
For testing set, data from test_matrix: with the number of pictures considered:
0: 0.9928571428571429
1: 0.9973568281938326
2: 0.9641472868217055
3: 0.9643564356435643
4: 0.9674134419551935
5: 0.9663677130044843
6: 0.9874739039665971
7: 0.9669260700389105
8: 0.944558521560575
9: 0.9593657086223984
precision_score: 
0.9715597201945959

Process finished with exit code 0
'''

#question 2: enhance the training sets and change the pictures
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

#result
'''
mnist is imported
start to enhancing the training sets
在方向1完成3个像素的剪切
在方向1完成6个像素的剪切
在方向2完成3个像素的剪切
在方向2完成6个像素的剪切
在方向3完成3个像素的剪切
在方向3完成6个像素的剪切
在方向4完成3个像素的剪切
在方向4完成6个像素的剪切
The sets have been enhanced.
best_knc loaded
start fitting
fitting done
start predicting
predicting done
For testing set, data from test_matrix: with the number of pictures considered:
0: 0.9908163265306122
1: 0.9973568281938326
2: 0.9496124031007752
3: 0.9702970297029703
4: 0.960285132382892
5: 0.9607623318385651
6: 0.9853862212943633
7: 0.9630350194552529
8: 0.9435318275154004
9: 0.9603567888999008
precision_score: 
0.9692054359770097
'''