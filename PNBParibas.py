import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
import gc

data = pd.read_csv('C:/Users/oussama/Documents/PNBParibas/train.csv',header =0)
def trait_messing_values(data):
    for k in data.columns :
        if(data.loc[:,k].dtype == np.dtype('float64')):
            data.loc[data.loc[:,k].isnull(),k]=data.loc[:,k].mean()
        else:
            data.loc[data.loc[:,k].isnull(),k] = data.loc[:,k].dropna().mode().values
trait_messing_values(data)
y = data.target.values
data.drop('target',axis=1,inplace=True)
def turn_dumies(data):
    for k in data.columns:
        if (data.loc[:,k].dtype ==np.dtype('float64')):
            continue
        else:
            data.loc[:,k] = pd.factorize(data.loc[:,k])[0]
turn_dumies(data)
X = data.values
#start modeling
xg = xgb.XGBClassifier(learning_rate=0.01)
rf=RandomForestClassifier(n_estimators=100,max_depth=3)
extra=ExtraTreesClassifier(n_estimators=100,max_depth=3)
models = [xg,extra,rf]
split = StratifiedKFold(y,n_folds=3)
for k , (train,test) in enumerate(split):
    X_train,X_test , y_train , y_test = X[train] , X[test] , y[train] , y[test]
    prediction_data= np.zeros((len(y_test),len(models)))
    j=0
    for model in models :
        model.fit(X_train,y_train)
        prediction_data[:,j] = model.predict_proba(X_test)[:,1]
        j +=1
    mean_prediction=np.mean(prediction_data,axis=1)
    print('the log loss of the {0} iteration is : {1}'.format(k,log_loss(y_test,mean_prediction)))

                                      
