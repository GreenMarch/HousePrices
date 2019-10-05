from pyspark.sql import *
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from pyspark.sql.functions import substring

import pandas as pd
import numpy as np

# data processing
# import statsmodels.api as sm
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sklearn import preprocessing
import sklearn.preprocessing as pp

import pandas as pd
# https://xgboost.readthedocs.io/en/latest/python/python_intro.html
import xgboost as xgb
print("xgboost", xgb.__version__)
from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier, XGBRegressor, XGBModel
from xgboost import plot_importance
# from matplotlib import pyplot
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
#from sklearn.metrics import sort

from sklearn.metrics import explained_variance_score
# from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

import datetime

from sklearn.model_selection import GridSearchCV

# import findspark
# findspark.init()
import pyspark # Call this only after findspark.init()
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# import seaborn as sns
# sns.set(style="darkgrid")
#
imp_out = []



# data prep
path = "/Users/lcui/study_toy_examples/house-prices-advanced-regression-techniques/house-prices-advanced-regression-techniques/train.csv"
df = pd.read_csv(path,header=0,sep=",")

cat_features = [a for a in ["Condition1",
"Condition2",
"LandContour",
"LandSlope",
"LotConfig",
"LotShape",
"MSSubClass","MSZoning",
"Neighborhood",
"SaleCondition",
"Alley",
"BldgType",
"BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","BsmtQual",
"CentralAir",
"Electrical","ExterCond",
"Exterior1st","Exterior2nd","ExterQual",
"Fence",
"FireplaceQu","Foundation","Functional",
"GarageCond","GarageFinish","GarageQual","GarageType",
"Heating","HeatingQC",
"HouseStyle",
"KitchenQual",
"MasVnrType",
"MiscFeature",
"PavedDrive",
"PoolQC",
"RoofMatl","RoofStyle",
"SaleType",
"Street",
"Utilities",
"GarageYrBlt","YearBuilt","YearRemodAdd"]]

num_features = [a for a in ["LotArea","LotFrontage",
"1stFlrSF","2ndFlrSF","3SsnPorch","BedroomAbvGr","BsmtFinSF1","BsmtFinSF2","BsmtFullBath","BsmtHalfBath","BsmtUnfSF",
"EnclosedPorch",
"Fireplaces","FullBath","GarageArea",
"GarageCars","GrLivArea",
"HalfBath",
"KitchenAbvGr","LowQualFinSF",
"MasVnrArea",
"MiscVal",
"MoSold","OpenPorchSF",
"OverallCond","OverallQual",
"PoolArea",
"ScreenPorch","TotalBsmtSF",
"TotRmsAbvGrd","WoodDeckSF"]]

for col in df.columns.tolist():
    print("col "
          + str(col)
          + " missing value -- "
          + str(bool(df[col].isnull().values.any())))


df[cat_features].shape # (1460, 47)
df_cat_onehot = pd.get_dummies(df[cat_features].astype(str).fillna("unknown")) #.values
df_cat_onehot.shape # (1460, 554)

df_num = df[num_features].fillna(0)

df_num['LotFrontage'] = df_num['LotFrontage'].fillna(0)
df_num = df_num.astype("float")
df_num.dtypes
# df_num = df_num.astype("int") #.values
columns_df_num = df_num.columns.tolist()

value_df_num = df_num.values#returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
value_df_num_scaled = min_max_scaler.fit_transform(value_df_num)
# value_df_num_scaled_remove_nan = np.nan_to_num(value_df_num_scaled) # didn't work
df_num_min_max_scaled = pd.DataFrame(value_df_num_scaled, columns=columns_df_num).fillna(0)


pt = pp.PowerTransformer(method='box-cox', standardize=False)
X_lognormal = df_num + 1
print("status 2e")
X_lognormal_array = pt.fit_transform(X_lognormal)
len(X_lognormal_array[1])
df_num_boxcox = pd.DataFrame(X_lognormal_array, columns=df_num.columns)

df_X_combine = pd.concat([df_num_min_max_scaled, df_cat_onehot],axis=1)

col_count = df.shape[1]
y = df.iloc[:,col_count-1]

df_X_Y_combine = pd.concat([df_X_combine, y],axis=1)
path = "/Users/lcui/study_toy_examples/house-prices-advanced-regression-techniques/house-prices-advanced-regression-techniques/X_Y_combine_min_max.csv"
df_X_Y_combine.to_csv(path,
                      sep=",",
                      header=True)


data = pd.read_csv(
path,
sep=",",
header=0) \
.drop("Unnamed: 0",axis=1) \
.fillna(0)

col_count = data.shape[1]
y_index = col_count - 1
print(y_index)
data_col_new = data.columns.str.replace(',', 'comma', regex=True)
data_col_new = data_col_new.str.replace('(', '_', regex=True)
data_col_new = data_col_new.str.replace(')', '_', regex=True)
data_col_new = data_col_new.str.replace('[', '_', regex=True)
data_col_new = data_col_new.str.replace(']', '_', regex=True)

data = np.nan_to_num(data)
data_new = pd.DataFrame(data,columns=data_col_new)


# exploration

X = data_new.iloc[:,:col_count-1].values.astype("float")
y = data_new.iloc[:,col_count-1].values.astype("float")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# grid search

start_time = datetime.datetime.now()

parameters = {
'learning_rate': [0.5,0.1],
'max_depth': [3,5,7,10],
'n_estimators': [50,75,100],
'objective': ['reg:squarederror'],
'random_state': [42],
'reg_alpha': [0.1, 0.5, 1],
'reg_lambda': [0.1, 0.5, 1],
'scale_pos_weight': [1],
'seed': [42],
'silent': [None],
'subsample': [1],
'verbosity': [1]}


# 2
xgb_reg = xgb.XGBRegressor()

# 3
grid_search = GridSearchCV(estimator=xgb_reg, cv=3, error_score='raise-deprecating',
            iid='warn', n_jobs=-1,
            param_grid=parameters,
            pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
            scoring=None, verbose=0)


print("all parameters:")
print(parameters)
# 4
grid_search.fit(X_train, y_train)

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")

best_parameters=grid_search.best_estimator_.get_params()
print("best params:")
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# best params:
# >>> for param_name in sorted(parameters.keys()):
# ...     print("\t%s: %r" % (param_name, best_parameters[param_name]))
# ...
# learning_rate: 0.1
# max_depth: 3
# n_estimators: 100
# objective: 'reg:squarederror'
# random_state: 42
# reg_alpha: 1
# reg_lambda: 1
# scale_pos_weight: 1
# seed: 42
# silent: None
# subsample: 1
# verbosity: 1

end_time = datetime.datetime.now()
print ('Select Done..., Time Cost: %d' % ((end_time - start_time).seconds))
############ feature imp
print(grid_search.best_estimator_.feature_importances_)
imp = grid_search.best_estimator_.feature_importances_ # np.sort(grid_search.best_estimator_.feature_importances_)
imp
print(len(imp_out))
imp_out.append(imp)
print(len(imp_out))


# OverallQual 0.23656
# BsmtQual 0.09060
# ExterQual 0.08669
# GarageCars 0.07652
# FullBath 0.07343
# KitchenQual 0.07099
# GarageType 0.05757
# GrLivArea 0.03118
# MSSubClass 0.02322
# TotalBsmtSF 0.01804
# TotRmsAbvGrd 0.01535
# CentralAir 0.01430
# Neighborhood 0.01423
# BsmtFinSF1 0.01417
# MSZoning 0.01253
# Fireplaces 0.01135
# 1stFlrSF 0.01126
# BsmtFinType1 0.01052
# HeatingQC 0.01013
# LandContour 0.00931
# 2ndFlrSF 0.00906
# Foundation 0.00722
# YearBuilt 0.00684
# SaleType 0.00642
# Exterior1st 0.00582
# SaleCondition 0.00566
# BsmtExposure 0.00536
# Condition1 0.00479
# YearRemodAdd 0.00474
# LotArea 0.00449
# LotShape 0.00400
# GarageYrBlt 0.00350
# WoodDeckSF 0.00292
# OverallCond 0.00292
# BsmtFinSF2 0.00275
# GarageArea 0.00263
# PoolArea 0.00262
# GarageFinish 0.00258
# ScreenPorch 0.00256
# OpenPorchSF 0.00248
# BsmtFullBath 0.00239
# MasVnrType 0.00235
# LotConfig 0.00208
# BedroomAbvGr 0.00202
# LotFrontage 0.00182
# Functional 0.00165
# KitchenAbvGr 0.00150
# LowQualFinSF 0.00122
# BsmtCond 0.00116
# BsmtUnfSF 0.00111
# HalfBath 0.00107
# FireplaceQu 0.00088
# HouseStyle 0.00071
# EnclosedPorch 0.00069
# MoSold 0.00065
# 3SsnPorch 0.00065
# PavedDrive 0.00040
# Exterior2nd 0.00030
# MasVnrArea 0.00008
# Alley 0.00000
# BldgType 0.00000
# BsmtFinType2 0.00000
# BsmtHalfBath 0.00000
# Condition2 0.00000
# Electrical 0.00000
# ExterCond 0.00000
# Fence 0.00000
# GarageCond 0.00000
# GarageQual 0.00000
# Heating 0.00000
# LandSlope 0.00000
# MiscFeature 0.00000
# MiscVal 0.00000
# PoolQC 0.00000
# RoofMatl 0.00000
# RoofStyle 0.00000
# SalePrice 0.00000
# Street 0.00000
# Utilities 0.00000


##############################################################################################


# model = grid_search.best_estimator_

best_parameters = {
'learning_rate': [0.5],
'max_depth': [3],
'n_estimators': [100],
'objective': ['reg:squarederror'],
'random_state': [42],
'reg_alpha': [1],
'reg_lambda': [1],
'scale_pos_weight': [1],
'seed': [42],
'silent': [None],
'subsample': [1],
'verbosity': [1]}


model = XGBRegressor(best_parameters)
model.fit(X_train, y_train)

# preds_train_best = model.predict(X_train)
preds_train = grid_search.predict(X_train)

#df_pred_true_y_train = pd.DataFrame({"y_train":y_train, "preds_train":preds_train, "preds_train_best":preds_train_best})
df_pred_true_y_train = pd.DataFrame({"y_train":y_train, "preds_train":preds_train})

correlation_train = df_pred_true_y_train.corr(method='pearson')
print(correlation_train)
#               y_train  preds_train
# y_train      1.000000     0.982068
# preds_train  0.982068     1.000000

#preds_best = model.predict(X_test)
preds = grid_search.predict(X_test)

print(y_test)
# [154500. 325000. 115000. 159000. 315500.  75500. 311500. 146000.  84500.
#  135500. 145000. 130000.  81000. 214000. 181000. 134500. 183500. 135000.
#  118400. 226000. 155000. 210000. 173500. 129000. 192000. 153900. 181134.
#  141000. 181000. 208900. 127000. 284000. 200500. 135750. 255000. 140000.
#  138000. 219500. 310000.  97000. 114500. 205000. 119500. 253293. 128500.
#  117500. 115000. 127000. 451950. 144000. 119000. 196000. 115000. 287000.
#  144500. 260000. 213000. 175000. 107000. 107500.  68500. 154000. 317000.
#  264132. 283463. 243000. 109000. 305000.  93500. 176000. 118858. 134000.
#  109008.  93500. 611657. 173000. 348000. 341000. 141000. 124900. 118000.
#   67000. 113000.  91300. 149500. 133000. 266000. 190000. 155900. 155835.
#  153500. 152000. 124500. 301000. 136500. 169990. 205000. 183900. 204900.
#  260000. 163500. 224900. 244000. 132000. 194000. 156500. 156000. 275000.
#  145000. 135000.  60000. 124000. 127000. 137500. 213500. 119000. 107900.
#  123000. 112000. 284000. 133000. 149000. 169000. 207000. 175000. 137000.
#  236000.  79500. 144000. 162900. 185900. 369900. 197900. 104000.  35311.
#  337500. 367294. 130250. 230000. 755000. 403000. 132000. 178000. 136500.
#  145000. 123000. 250000. 187100. 133900.  67000. 137500. 155000. 200624.
#  154300.  91000. 136000. 108959. 140000.  86000. 131400. 179900. 144000.
#  293077. 144500. 118500. 141000. 239000. 276000. 556581. 244400. 360000.
#  103200. 102000. 151000. 285000. 134432. 113000. 187500. 125500. 177500.
#  179900.  55993. 132500. 135000. 255000. 140000. 271000. 246578. 202500.
#   75000. 122500. 108480. 160000. 171000. 196000. 225000. 197000.  40000.
#  172500. 154900. 280000. 175000. 147000. 315000. 185000. 135500. 239500.
#  139000. 140000. 110000. 225000. 143500. 128950. 172500. 241500. 262500.
#  194201. 143000. 130000. 126000. 142500. 254000. 217500.  66500. 201000.
#  155000.  68400.  64500. 173000. 102776.  84900. 165600. 120000. 135000.
#  220000. 153575. 195400. 147000. 277000. 143000. 105900. 242000. 194500.
#  438780. 185000. 107500. 165000. 176000. 129900. 115000. 192140. 160000.
#  145000.  86000. 158000. 127500. 115000. 119500. 175900. 240000. 395000.
#  165000. 128200. 275000. 311872. 214000. 153500. 144000. 115000. 180000.
#  465000. 180000. 253000.  85000. 101800. 148500. 137500. 318061. 143000.
#  140000. 192500.  92000. 197000. 109500. 297000. 185750. 230000.  89471.
#  260000. 189000. 108000. 124500.]

print(preds)
# [140187.47  332030.44  120576.45  160727.83  302323.38   83274.
#  221520.16  149448.97   83697.43  138744.48  159403.69  125968.53
#  111023.26  199437.69  169355.4   137405.73  193344.61  135368.22
#  117396.46  211307.03  141995.47  218428.55  168149.88  121251.25
#  197931.69  160733.25  198857.28  110855.79  175815.66  199458.3
#  129575.02  251394.84  231019.25  112706.31  244125.81  141642.22
#  127098.9   206812.75  327736.06  107827.836 130596.74  238669.25
#  116806.4   387010.66  126451.28  139120.92  116648.89  130193.72
#  422829.97  130349.95  127946.73  211508.3   106738.32  370983.03
#  146138.66  238070.14  198600.11  155732.89  136325.34  106574.34
#   71850.18  165687.55  316622.97  304061.6   293031.66  218728.55
#  108730.77  315874.16  112701.4   151439.25  126552.67  126692.49
#  111360.555  87722.414 366046.22  183475.02  315867.75  298025.9
#  140476.6   113258.84  105167.46   83694.664 124474.33   93288.61
#  149015.33  127868.29  262030.66  206065.61  144792.    183626.52
#  140351.08  138090.1   123082.87  266529.84  114967.266 181758.27
#  182342.56  168433.66  210686.61  243252.25  186542.75  205560.16
#  271885.88  137741.36  194697.19  149764.8   145519.84  274807.3
#  134053.47  180175.02   60019.344 122479.6   135375.08  125841.64
#  197736.78  120725.8   103661.12  111243.61  139031.97  267428.6
#  152076.33  144792.    178083.47  184944.95  177076.14  134140.28
#  232431.25  105589.91  141520.45  187624.17  195514.45  381280.75
#  195247.06  126080.59   63714.215 325278.34  378311.9   125784.76
#  222944.72  599294.5   384210.8   135472.88  175301.52  167062.27
#  155335.27  128926.664 235410.98  195767.31  117632.92   68266.805
#  116788.71  144612.44  258131.1   155924.77   97816.234 127734.85
#  141641.95  151328.67   95358.6   133445.8   222377.52  141729.44
#  299790.22  138738.3   110775.445 113791.195 210406.02  368728.28
#  449374.4   230798.67  370705.7    96032.11  120833.664 157388.14
#  318225.5   132280.52  129788.12  203028.94  118630.33  166143.95
#  177015.61  108088.58  124063.46  140374.44  261633.1   156616.64
#  293439.94  220563.73  190984.12   86191.47  112208.15  108510.99
#  139266.31  148205.89  177544.52  162274.8   230502.61   91171.32
#  211478.75  124090.46  240763.08  201776.73  122765.76  336511.6
#  191734.64  124570.25  246431.03  141329.95  149708.45  118421.62
#  241051.08  139287.62  116721.96  145066.05  219299.55  257528.92
#  188255.67  139785.98  118462.65  136612.08  145082.38  234600.28
#  208977.78   94816.125 236190.61  149871.55   90645.21   99517.74
#  169695.34  106883.97  105967.445 179036.88  128978.836 139562.86
#  234044.45  128883.555 194930.42  152010.08  237462.36  133231.3
#  108824.836 241408.11  203750.12  398278.75  190719.4   135623.73
#  152012.44  170699.16  156099.78   95992.75  163518.53  174228.83
#  131594.16   98328.28  137159.31  146591.88  104539.89  104165.98
#  173001.72  270475.94  290101.4   166387.8   125883.984 229056.55
#  275216.44  201570.02  163800.39  138477.25  116193.38  165111.45
#  413961.34  226052.61  220673.61   88072.29  108749.31  128324.85
#  153565.25  307060.78  220106.72  137455.78  199821.83  108458.35
#  201679.83  113003.81  318109.47  186252.66  216652.69  107959.33
#  233137.19  183622.4   114706.48  116014.875]

#df_pred_true_y = pd.DataFrame({"y_test":y_test, "preds":preds, "preds_best":preds_best})
df_pred_true_y = pd.DataFrame({"y_test":y_test, "preds":preds})
correlation_test = df_pred_true_y.corr(method='pearson')
print(correlation_test)
#           y_test     preds
# y_test  1.000000  0.944287
# preds   0.944287  1.000000
# exploration end


# new draft
xgb = model

feature_names = data_new.columns[:-1].to_list().values
feature_names

# use DMatrix for xgbosot
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

# params = {'nestimators':[5000], 'learningrate':[0.6], 'njobs': [4], 'silent': [False], 'objective': ['gpu:reg:linear'], 'maxdeltastep': [1], 'gamma': [0.5], 'randomstate': [42]}

# grid = GridSearchCV(xgb, params)


# grid.fit(X_train, y_train, verbose=True)
# cv_results = grid.cv(dtrain=data_dmatrix, params=params, nfold=3,
#                     num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


# use svmlight file for xgboost
dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('dtrain.svm')
dtest_svm = xgb.DMatrix('dtest.svm')

# set xgboost params
param = {
    'max_depth': 5,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'reg:squarederror', # multi:softprob',  # error evaluation for multiclass training
    #'num_class': 2 # the number of classes that exist in this datset
    }  
num_round = 20  # the number of training iterations




# https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
# class xgboost.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, verbosity=1, silent=None, objective='reg:squarederror', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None, importance_type='gain', **kwargs)
#------------- numpy array ------------------
# training and testing - numpy matrices
bst = xgb.train(param, dtrain, num_round)
bst.fit(X_train,y_train) # AttributeError: 'Booster' object has no attribute 'fit'

# preds = xg_reg.predict(X_test)
preds = bst.predict(dtest)

importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
for f in importance_types:
bst.get_score(importance_type=f)

from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

explained_variance_score(y_test, preds)
max_error(y_test, preds)
mean_absolute_error(y_test, preds)
mean_squared_error(y_test, preds)

median_absolute_error(y_test, preds)
r2_score(y_test, preds)



# # extracting most confident predictions
# best_preds = np.asarray([np.argmin(line) for line in preds])
# print ("Numpy array precision:", mean_squared_error(y_test, best_preds))

# ------------- svm file ---------------------
# training and testing - svm file
bst_svm = xgb.train(param, dtrain_svm, num_round)
preds = bst.predict(dtest_svm)

# extracting most confident predictions
best_preds_svm = [np.argmax(line) for line in preds]
print ("Svm file precision:",mean_squared_error(y_test, best_preds_svm))
# --------------------------------------------

# dump the models
bst.dump_model('dump.raw.txt')
bst_svm.dump_model('dump_svm.raw.txt')


# save the models for later
joblib.dump(bst, 'bst_model.pkl', compress=True)
joblib.dump(bst_svm, 'bst_svm_model.pkl', compress=True)

# load model from file
loaded_model = joblib.load("bst_model.pkl")



# # Re-run your model with fit, save, and then load pipeline file like normal.
# # Save Model
# joblib.dump(bst, "xgb1.joblib.dat")
# # load model from file
# loaded_model = joblib.load("xgb1.joblib.dat")

# # Available importance_types = [‘weight’, ‘gain’, ‘cover’, ‘total_gain’, ‘total_cover’]
# f = 'weight'
# # f = 'gain'
# # f = 'cover'
# # f = 'total_gain'
# # f = 'total_cover'

# # loaded_model = XGBClassifier().get_booster().get_score(importance_type=f)
# # loaded_model.fit(X, y)

# loaded_model = XGBRegressor().get_booster().get_score(importance_type=f)
loaded_model.fit(X, y)

# plot feature importance
plot_importance(loaded_model)
pyplot.show()
