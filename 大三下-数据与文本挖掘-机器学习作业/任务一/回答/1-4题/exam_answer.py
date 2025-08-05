#附着在运行环境（模板）后，在命令行条件下运行
#！！！该文件只用于记录代码以及运行结果，无法直接运行！！！
#关于实际运行的程序，需参见题目相应的文件
#question 1: sklearn.svm.SVR
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR
#1.kernel='rbf'
svm_work = SVR(gamma='auto', cache_size=500, verbose=True)

svm_work.fit(housing_prepared, housing_labels)
housing_predict = svm_work.predict(housing_prepared)
mse = mean_squared_error(housing_labels, housing_predict)
rmse = np.sqrt(mse)
print('rbf:')
print(rmse)

#result
#[LibSVM]118577.43356412371

#2.kernel='linear'
svm_work = SVR(kernel='linear', gamma='auto', cache_size=500, verbose=True)

svm_work.fit(housing_prepared, housing_labels)
housing_predict = svm_work.predict(housing_prepared)
mse = mean_squared_error(housing_labels, housing_predict)
rmse = np.sqrt(mse)
print('linear')
print(rmse)

#result
#[LibSVM]111094.6308539982

#3.kernel='poly'
svm_work = SVR(kernel='poly', degree=5, gamma='auto', cache_size=500, verbose=True)

svm_work.fit(housing_prepared, housing_labels)
housing_predict = svm_work.predict(housing_prepared)
mse = mean_squared_error(housing_labels, housing_predict)
rmse = np.sqrt(mse)
print('poly')
print(rmse)

#result
#[LibSVM]118548.12154995742

#4.kernel='sigmoid'
svm_work = SVR(kernel='sigmoid', gamma='auto', cache_size=500, verbose=True)

svm_work.fit(housing_prepared, housing_labels)
housing_predict = svm_work.predict(housing_prepared)
mse = mean_squared_error(housing_labels, housing_predict)
rmse = np.sqrt(mse)
print('sigmoid:')
print(rmse)

#result
#[LibSVM]118427.91887687655

#5.kernel='precomputed'
#无法用现有数据检验："Precomputed matrix must be a square matrix."

#命令行结果
'''
[LibSVM].........
Warning: using -h 0 may be faster
*
optimization finished, #iter = 9987
obj = -1456651684.858144, rho = -179534.616242
nSV = 16512, nBSV = 16510
rbf:
118577.43356412371
[LibSVM].........
Warning: using -h 0 may be faster
*
optimization finished, #iter = 9994
obj = -1403378590.128254, rho = -179692.610605
nSV = 16512, nBSV = 16510
linear
111094.6308539982
[LibSVM].............................*...........................................*
optimization finished, #iter = 72073
obj = -1454677477.998141, rho = -179310.497665
nSV = 16512, nBSV = 16484
poly
118548.12154995742
[LibSVM]..........
Warning: using -h 0 may be faster
*
optimization finished, #iter = 10544
obj = -1455918635.336038, rho = -179486.252008
nSV = 16512, nBSV = 16512
sigmoid:
118427.91887687655

Process finished with exit code 0
'''

#comparison
'''
rbf: 118577.43356412371
linear: 111094.6308539982
poly: 118548.12154995742
sigmoid: 118427.91887687655

综上，在"degree=5, gamma='auto', cache_size=500, verbose=True"的条件下，预测效果：
linear > sigmoid > poly > rbf

于是最佳的SVR是kernel='linear'，其表现为rmse = 111094.6308539982
'''

#question 2: GridSearchCV --> RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
import joblib
#prepare
for_reg = RandomForestRegressor(n_estimators=30, n_jobs=-1, random_state=64)
pa_dict = {
    'n_estimators': randint(low=1, high=50),
    'max_features': randint(low=1, high=10),
}
ran_cv = RandomizedSearchCV(for_reg, pa_dict, n_jobs=-1, scoring='neg_mean_squared_error', random_state=64, n_iter=25, cv=4, return_train_score=True)
#fit
print('start fitting')
ran_cv.fit(housing_prepared, housing_labels)
print('fitting done')
#save the model
print('saving the model')
joblib.dump(ran_cv, './ran_cv.pkl')
print('model saved')
#show
print(ran_cv.best_params_)
cv_result = ran_cv.cv_results_
for score, param in zip(cv_result['mean_test_score'], cv_result['params']):
    print(np.sqrt(-score), param)

#命令行结果
'''
start fitting
fitting done
saving the model
model saved
{'max_features': 8, 'n_estimators': 45}
50338.90255463702 {'max_features': 5, 'n_estimators': 39}
50050.71219223338 {'max_features': 8, 'n_estimators': 39}
50625.47093215319 {'max_features': 7, 'n_estimators': 22}
55660.828884688206 {'max_features': 1, 'n_estimators': 32}
52320.07045883298 {'max_features': 9, 'n_estimators': 11}
50465.974056598265 {'max_features': 7, 'n_estimators': 30}
50218.25753697434 {'max_features': 9, 'n_estimators': 35}
55729.7119201235 {'max_features': 1, 'n_estimators': 31}
60637.70630110698 {'max_features': 5, 'n_estimators': 3}
50748.216735785936 {'max_features': 7, 'n_estimators': 21}
50276.97094290539 {'max_features': 6, 'n_estimators': 28}
58568.89731458198 {'max_features': 1, 'n_estimators': 11}
50311.93294335434 {'max_features': 6, 'n_estimators': 26}
50109.91642586417 {'max_features': 7, 'n_estimators': 45}
51501.85184513349 {'max_features': 3, 'n_estimators': 30}
51248.57441152715 {'max_features': 7, 'n_estimators': 15}
53536.97710318906 {'max_features': 3, 'n_estimators': 12}
50548.403795695594 {'max_features': 4, 'n_estimators': 44}
52987.87336561623 {'max_features': 9, 'n_estimators': 9}
49951.753763096065 {'max_features': 8, 'n_estimators': 45}
55365.085612038696 {'max_features': 1, 'n_estimators': 37}
50073.879033647376 {'max_features': 6, 'n_estimators': 35}
50197.08178092244 {'max_features': 7, 'n_estimators': 42}
51233.74160294379 {'max_features': 6, 'n_estimators': 16}
50570.269269661345 {'max_features': 4, 'n_estimators': 38}

Process finished with exit code 0
'''

#question 3: pipeline
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import joblib
#Firstly, use the ran_cv.pkl in question 2 to get feature_importances
ran_cv = joblib.load('./ran_cv.pkl')
print('model loaded')
f_importance = ran_cv.best_estimator_.feature_importances_
np_f_import = np.array(f_importance)
print('feature importance loaded:')
print(np_f_import)

#select the number of the most important attributes to keep
num_att = 4
indice_k_top = np.sort(np.argpartition(np_f_import,-num_att)[-num_att:])
print("the indice of the top k elements:")
print(indice_k_top)
print("their names are:")
print(np.array(attributes)[indice_k_top])
print(sorted(zip(f_importance, attributes), reverse=True)[:num_att])

#define a class for convenience
from sklearn.base import BaseEstimator, TransformerMixin
class TopSelector(BaseEstimator, TransformerMixin):
    def __init__(self, f_importance, num_att):
        self.f_importance = f_importance
        self.num_att = num_att
    def fit(self, X, y=None):
        self.f_indices_ = np.sort(np.argpartition(self.f_importance,-self.num_att)[-self.num_att:])
        return self
    def transform(self, X):
        return X[:, self.f_indices_]

#built a new pipeline
new_selection_pipeline = Pipeline(
    [('preparation', full_pipeline), ('feature_selection', TopSelector(f_importance, num_att))]
)

#get the aimed housing_data
housing_top_features = new_selection_pipeline.fit_transform(housing)
print("housing_top_features[:]")
print(housing_top_features[:])
print("housing_top_features[indices_k_top]")
print(housing_top_features[indice_k_top])

#result
'''
model loaded
feature importance loaded:
[7.11599088e-02 6.47578081e-02 4.31021601e-02 1.57994338e-02
 1.46843381e-02 1.50156313e-02 1.43459181e-02 3.93443017e-01
 4.56982779e-02 1.11153941e-01 4.73289237e-02 4.57348921e-03
 1.52958784e-01 1.10066541e-04 2.80402286e-03 3.06427975e-03]
the indice of the top k elements:
[ 0  7  9 12]
their names are:
['longitude' 'median_income' 'pop_per_hhold' 'INLAND']
[(0.3934430170603441, 'median_income'), (0.15295878373404143, 'INLAND'), (0.11115394099013248, 'pop_per_hhold'), (0.0711599088433022, 'longitude')]
housing_top_features[:]
[[-1.15604281 -0.61493744 -0.08649871  0.        ]
 [-1.17602483  1.33645936 -0.03353391  0.        ]
 [ 1.18684903 -0.5320456  -0.09240499  0.        ]
 ...
 [ 1.58648943 -0.3167053  -0.03055414  1.        ]
 [ 0.78221312  0.09812139  0.06150916  0.        ]
 [-1.43579109 -0.15779865 -0.09586294  0.        ]]
housing_top_features[indices_k_top]
[[-1.15604281 -0.61493744 -0.08649871  0.        ]
 [ 1.16686701  1.11523946 -0.05183121  1.        ]
 [ 0.64733449 -1.77211596  0.80278786  0.        ]
 [-1.55568321  1.4750499  -0.02129013  0.        ]]

Process finished with exit code 0
'''

#question 4: GridSearchCV
#注：这个程序跑得非常慢，我的电脑用了十多分钟
#import pipeline in question 3
from sklearn.base import BaseEstimator, TransformerMixin
class TopSelector(BaseEstimator, TransformerMixin):
    def __init__(self, f_importance, k):
        self.f_importance = f_importance
        self.k = k
    def fit(self, X, y=None):
        self.f_indices_ = np.sort(np.argpartition(self.f_importance,-self.k)[-self.k:])
        return self
    def transform(self, X):
        return X[:, self.f_indices_]

#built a new pipeline
import  joblib
print('loading model')
ran_cv = joblib.load('./ran_cv.pkl')
print('model loaded')
f_importances = ran_cv.best_estimator_.feature_importances_
k = 4
new_selection_pipeline = Pipeline(
    [('preparation', full_pipeline), ('feature_selection', TopSelector(f_importances, k))]
)
print('pipeline loaded')

#教材基础步：开始
#rnd_search
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from scipy.stats import expon, reciprocal
svm_reg = SVR()
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=5, cv=5, scoring='neg_mean_squared_error',
                                verbose=2, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)


predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopSelector(f_importances, k)),
    ('svm_reg', SVR(**rnd_search.best_params_))
])
#教材基础步：结束

#begin
from sklearn.model_selection import GridSearchCV
para = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(f_importances) + 1))
}]
grid_search = GridSearchCV(predict_pipeline, para, n_jobs=-1, refit=True, scoring='neg_mean_squared_error', return_train_score=True)
print('start fitting')
grid_search.fit(housing, housing_labels)
print('fitting done')
print('grid_search.best_estimator_ :')
print(grid_search.best_estimator_)
print('grid_search.best_score_ :')
print(grid_search.best_score_)
print('grid_search.best_index_ :')
print(grid_search.best_index_)

#result
'''
loading model
model loaded
pipeline loaded
Fitting 5 folds for each of 5 candidates, totalling 25 fits
[CV] END C=629.782329591372, gamma=3.010121430917521, kernel=linear; total time=   6.0s
[CV] END C=629.782329591372, gamma=3.010121430917521, kernel=linear; total time=   6.1s
[CV] END C=629.782329591372, gamma=3.010121430917521, kernel=linear; total time=   6.1s
[CV] END C=629.782329591372, gamma=3.010121430917521, kernel=linear; total time=   6.3s
[CV] END C=629.782329591372, gamma=3.010121430917521, kernel=linear; total time=   6.3s
[CV] END C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf; total time=  11.4s
[CV] END C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf; total time=  11.6s
[CV] END C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf; total time=  11.3s
[CV] END C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf; total time=  11.4s
[CV] END C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf; total time=  11.6s
[CV] END C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf; total time=   9.2s
[CV] END C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf; total time=   9.5s
[CV] END C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf; total time=   9.5s
[CV] END C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf; total time=   9.5s
[CV] END C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf; total time=   9.5s
[CV] END C=432.37884813148855, gamma=0.15416196746656105, kernel=linear; total time=   6.2s
[CV] END C=432.37884813148855, gamma=0.15416196746656105, kernel=linear; total time=   6.2s
[CV] END C=432.37884813148855, gamma=0.15416196746656105, kernel=linear; total time=   6.3s
[CV] END C=432.37884813148855, gamma=0.15416196746656105, kernel=linear; total time=   6.3s
[CV] END C=432.37884813148855, gamma=0.15416196746656105, kernel=linear; total time=   6.3s
[CV] END C=24.17508294611391, gamma=3.503557475158312, kernel=rbf; total time=  10.3s
[CV] END C=24.17508294611391, gamma=3.503557475158312, kernel=rbf; total time=  10.9s
[CV] END C=24.17508294611391, gamma=3.503557475158312, kernel=rbf; total time=  10.3s
[CV] END C=24.17508294611391, gamma=3.503557475158312, kernel=rbf; total time=  10.3s
[CV] END C=24.17508294611391, gamma=3.503557475158312, kernel=rbf; total time=  10.5s
start fitting
fitting done
grid_search.best_estimator_ :
Pipeline(steps=[('preparation',
                 ColumnTransformer(transformers=[('num',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer()),
                                                                  ('attribs_adder',
                                                                   CombinedAttributesAdder()),
                                                                  ('std_scaler',
                                                                   StandardScaler())]),
                                                  ['longitude', 'latitude',
                                                   'housing_median_age',
                                                   'total_rooms',
                                                   'total_bedrooms',
                                                   'population', 'households',
                                                   'median_income']),
                                                 ('cat', OneHotEncoder(),
                                                  ['ocean_proximi...
                ('feature_selection',
                 TopSelector(f_importance=array([7.11599088e-02, 6.47578081e-02, 4.31021601e-02, 1.57994338e-02,
       1.46843381e-02, 1.50156313e-02, 1.43459181e-02, 3.93443017e-01,
       4.56982779e-02, 1.11153941e-01, 4.73289237e-02, 4.57348921e-03,
       1.52958784e-01, 1.10066541e-04, 2.80402286e-03, 3.06427975e-03]),
                             k=8)),
                ('svm_reg',
                 SVR(C=26290.206464300216, gamma=0.9084469696321253))])
grid_search.best_score_ :
-3582318395.0126357
grid_search.best_index_ :
21

Process finished with exit code 0
'''