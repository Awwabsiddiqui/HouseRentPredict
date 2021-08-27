import pandas as pd
from sklearn import preprocessing,linear_model
import numpy as np
import sklearn

data=pd.read_csv('houses_to_rent.csv',sep=',')
data=data[['city','rooms','bathroom','animal',
           'furniture','rent amount','fire insurance']]

data['rent amount'] = data['rent amount'].map(lambda i:int (i[2:].replace(',','')))
data['fire insurance'] = data['fire insurance'].map(lambda i:int (i[2:].replace(',','')))
le=preprocessing.LabelEncoder()
data['furniture']=le.fit_transform((data['furniture']))
data['animal']=le.fit_transform((data['animal']))

print (data.head())
print('-'*50)
x=np.array(data.drop(['rent amount'],1))
y=np.array(data['rent amount'])
print('X',x.shape)
print('Y',y.shape)

xtr,xt,ytr,yt=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
print('xtrain',xtr.shape)
print('xtest',xt.shape)

model=linear_model.LinearRegression()
model.fit(xtr,ytr)
acc=model.score(xt,yt)
print('COF',model.coef_)
print('Intrcept',model.intercept_)
print('acc',round(acc*100,3),'%')
tstval=model.predict(xt)
print('test value',tstval.shape)
error=[]
for i,tstval in enumerate(tstval):
    error.append(yt[i]-tstval)
    print(f'actual:{yt[i]}   predict:{int(tstval)}   error:{int(error[i])}')