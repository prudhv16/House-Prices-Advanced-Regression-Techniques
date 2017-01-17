'''
Author :- Prudhvi Indana
Project :- House_Prices_Advanced_Regression_Techniques
Dataset taken from :- https://www.kaggle.com/c/house-prices-advanced-regression-techniques
'''

'''Importing necessary packages for analysis.'''

#import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
#import matplotlib as mpl
#from sklearn import decomposition
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge    #, ElasticNet, Lasso, LassoLarsCV
from sklearn import svm
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import mean_squared_error

sns.set(color_codes=True)
#import scipy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pylab
scalar = MinMaxScaler()

#reading data from csv files for preprocessing

train_data = pd.read_csv("train.csv",index_col=0)
test_data = pd.read_csv("test.csv",index_col=0)
y_train = train_data["SalePrice"]
y_train = np.log1p(y_train)
#y_test = test_data[""]
del train_data["SalePrice"]
train_data['MSSubClass'] = train_data['MSSubClass'].astype(basestring)
data = pd.concat((train_data, test_data), axis=0)
#allColumns = data.columns	


#Removing missing values and replacing with mean values in all columns.
meancoldata = data.mean()
#meancoldata.head()
data = data.fillna(meancoldata)

#Taking first 20 columns for analysis.
data = data[["MSSubClass","MSZoning","LotArea","LandContour","Neighborhood",
             "BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt",
             "YearRemodAdd","Exterior1st","Exterior2nd","MasVnrArea","ExterQual",
             "Foundation","BsmtQual","BsmtCond","BsmtFinType1","BsmtFinSF1",
            "BsmtFinType2","BsmtUnfSF","TotalBsmtSF","HeatingQC","CentralAir",
             "1stFlrSF","2ndFlrSF","GrLivArea","BsmtFullBath","FullBath",
             "HalfBath","BedroomAbvGr","KitchenAbvGr","KitchenQual","TotRmsAbvGrd",
             "Functional","Fireplaces","FireplaceQu","GarageType","GarageYrBlt",
            "GarageFinish","GarageCars","GarageArea","GarageQual","GarageCond",
             "PavedDrive","WoodDeckSF","OpenPorchSF","Fence"]]
#datapart2 = data[['1stFlrSF','2ndFlrSF']]
data = data.iloc[:,:20]

corr = data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


#Befor plot of target variable
sns.distplot(y_train)
sns.plt.show()


#after plot of target variables
y_train = np.log1p(y_train)
sns.distplot(y_train)
sns.plt.show()


#Before plot of BsmtFinSF1 varaible
plt.rcParams['figure.figsize'] = (10.0, 4.0)
sns.distplot(data['BsmtFinSF1'])
sns.plt.show()


#Taking log for converting and plotting BsmtFinSF1 varaible
data['BsmtFinSF1'] = np.log1p(data['BsmtFinSF1'])
sns.distplot(data['BsmtFinSF1'])
sns.plt.show()


#Before plot of MasVnrArea varaible
sns.distplot(data['MasVnrArea'])
sns.plt.show()


#Taking log for converting and plotting MasVnrArea varaible
data['MasVnrArea'] = np.log1p(data['MasVnrArea'])
sns.distplot(data['MasVnrArea'])
sns.plt.show()


#Similarly before and after plots for varaibles 'LotArea'
sns.distplot(data['LotArea'])
data['LotArea'] = np.log1p(data['LotArea'])
sns.distplot(data['LotArea'])
sns.plt.show()


#Generating dummy vataibles for all categorical variables

X_train = data[:train_data.shape[0]]
test = data[train_data.shape[0]:]
X_train = pd.get_dummies(X_train,dummy_na=True)

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)

alpha_ridge = [1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 3, 5, 7, 10, 20, 30]
crossval_ridge = [np.mean(rmse_cv(Ridge(alpha=x,max_iter=1000))) for x in alpha_ridge]

crossval_ridge = pd.Series(crossval_ridge, index = alpha_ridge)
crossval_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")
pylab.show()

param_grid = [
  {'C': [1, 10, 50, 100, 500, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 50, 100, 500, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
 
reg = svm.SVR()
reg_cv = GridSearchCV(reg, param_grid,scoring='neg_mean_squared_error')
reg_cv.fit(X_train, y_train)

'''
Note:- I reported SVR choosing rbf kernal and gamma as 0.001 and c as 50, some how it tends to choose linear kernal with c as 1 best perameter now.

It could be issue with seed.
'''

print "Best score for SVR after grid search is ",reg_cv.best_score_ 
print "Best perameters choosen by grid search are ", reg_cv.best_params_

#Below code is needed for ploting SVR error for different values of C

SVC_C = [10,50,100,500,1000]
crossval_SVR = [np.mean(rmse_cv(svm.SVR(kernel='rbf',gamma=0.0001,C=x))) for x in SVC_C]
crossval_ridge = pd.Series(crossval_SVR, index = SVC_C)
crossval_ridge.plot(title = "Validation")
plt.xlabel("C in SVR")
plt.ylabel("rmse")
pylab.show()


#Code for PCA analysis
pca = PCA(n_components=5,svd_solver = 'full',random_state=16)
pca.fit(X_train)
print(pca.explained_variance_ratio_)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
pylab.show()

data_PCA = pca.transform(X_train)
#alpha_ridge = [0,1,2,3,4]
#crossval_ridge = [np.mean(rmse_cv_PCA(Ridge(alpha=5,max_iter=1000)))]


def rmse_cv_PCA(model,tempdata):
    '''
    model :- model to be passed for evaluation
    temdata :- data on on which model wiht 10 fold cross fold validation will be applied.
    '''
    rmse= np.sqrt(-cross_val_score(model, tempdata, y_train, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)
	
alpha_ridge = [1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 3, 5, 7, 10]
crossval_ridge1 = [np.mean(rmse_cv_PCA(Ridge(alpha=x,max_iter=1000),data_PCA[:,:1])) for x in alpha_ridge]
crossval_ridge2 = [np.mean(rmse_cv_PCA(Ridge(alpha=x,max_iter=1000),data_PCA[:,:2])) for x in alpha_ridge]
crossval_ridge3 = [np.mean(rmse_cv_PCA(Ridge(alpha=x,max_iter=1000),data_PCA[:,:3])) for x in alpha_ridge]
crossval_ridge4 = [np.mean(rmse_cv_PCA(Ridge(alpha=x,max_iter=1000),data_PCA[:,:4])) for x in alpha_ridge]
crossval_ridge5 = [np.mean(rmse_cv_PCA(Ridge(alpha=x,max_iter=1000),data_PCA[:,:5])) for x in alpha_ridge]

plt.plot(alpha_ridge, crossval_ridge1)
plt.plot(alpha_ridge, crossval_ridge2)
plt.plot(alpha_ridge, crossval_ridge3)
plt.plot(alpha_ridge, crossval_ridge4)
plt.plot(alpha_ridge, crossval_ridge5)
plt.xlabel('N dimesion')
plt.ylabel('RMSE')
plt.legend(['1st principal dimension', '1st+2nd principal dimensions', '1st+2nd+3rd principal dimensions', 
            '1st+2nd+3rd+4th principal dimensions','1st+2nd+3rd+4th+5th principal dimensions'], loc='upper left',
           borderaxespad=0.)
pylab.show()