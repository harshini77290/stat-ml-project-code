# stat-ml-project-code
# import libraries
import numpy as npa
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
# import evalml
# from evalml.automl import AutoMLSearch

! pip install openpyxl

## Loading Dataset
The first step is to load dataset into memory. We have been given flight fare prediction dataset that is being loaded below.

# load excel dataset & convert it into .csv file
raw = pd.read_excel("/content/drive/MyDrive/air/Data_Train.xlsx")
# export to csv
raw.to_csv('./train.csv')

raw_test = pd.read_excel("/content/drive/MyDrive/air/Test_set.xlsx")
raw_test.to_csv('./test.csv')

# load csv dataset
data = pd.read_csv('./train.csv')

test = pd.read_csv('./test.csv')

data.shape, test.shape

## EDA (Exploratory Data Analysis)

pd.set_option('display.max_columns', None)

data.head(4)

# lets remove the unnamed column from our dataset
data = data.drop('Unnamed: 0', axis=1)

data.head(4)

data.columns

cat = [i for i in data.columns if data[i].dtype == object]
num = [i for i in data.columns if data[i].dtype != object]

cat

data.shape

data['Date_of_Journey'].dtype

# pd.to_datetime(data['Date_of_Journey'].dtype)

data.isnull().sum().to_numpy()

data.dropna(inplace=True)

data.isnull().sum().to_numpy()

data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], 
                                         infer_datetime_format=True)

# data = data.drop('Unnamed: 0', axis=1)

data.head(4)

### Creating Additional Features (Feature Engineering)


data['Day of Journey'] = data['Date_of_Journey'].dt.day
data['Month of Journey'] = data['Date_of_Journey'].dt.month
data['Year of Journey'] = data['Date_of_Journey'].dt.year

data.head(3)

# after making additional features from datetime, now we remove the datetime feature 
data.drop('Date_of_Journey', axis=1, inplace=True)

data.head()

data['Year of Journey'].value_counts()

data['Dep_Time'] = pd.to_datetime(data['Dep_Time'], 
                                         infer_datetime_format=True)

data['Dep_Time'].dtype

data['Hour of Departure'] = data['Dep_Time'].dt.hour
data['Minute of Departure'] = data['Dep_Time'].dt.minute

# dropping original departure time feature
data.drop('Dep_Time', axis=1, inplace=True)

data.head(4)

data['Arrival_Time'] = pd.to_datetime(data['Arrival_Time'], 
                                         infer_datetime_format=True)

data['Arrival Hour'] = data.Arrival_Time.dt.hour
data['Arrival Minute'] = data.Arrival_Time.dt.minute

data.drop('Arrival_Time', axis=1, inplace=True)

data.head(3)

duration = list(data['Duration'])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + ' 0m'
        else:
            duration[i] = "0h " + duration[i]

duration_hrs = []
duration_mins = []

for i in range(len(duration)):
    duration_hrs.append(int(duration[i].split(sep='h')[0]))
    duration_mins.append(int(duration[i].split(sep='m')[0].split()[-1]))    

data['Duration Hours'] = duration_hrs
data['Duration Mins'] = duration_mins

data.head(2)

data.drop('Duration', axis=1, inplace=True)

data.head(2)

data['Total_Stops'].value_counts()

sns.set(rc={'figure.figsize':(10, 5)})
sns.countplot(x=data.Total_Stops)

sns.set(rc={'figure.figsize':(15, 10)})

sns.catplot(y='Price', x='Airline', data=data.sort_values("Price", ascending=False),
           kind='boxen', height=6, aspect=3)

### Converting Airline Named to One-hot Encoded Vectors

airline_names = data[['Airline']]
airline_names = pd.get_dummies(airline_names, drop_first=True)

airline_names.head(3)

sns.catplot(y='Price', x='Source', data=data.sort_values("Price", ascending=False),
           kind='boxen', height=6, aspect=3)

### Converting Source to One-hot Encoded Vectors

data['Source'].value_counts()

source = data[['Source']]
source = pd.get_dummies(source)

source.head(4)

data['Destination'].value_counts()

### Converting Destination to One-hot Encoded Vectors

destination = data[['Destination']]
destination = pd.get_dummies(destination)

destination.head(3)

destination.shape

data

data.drop(['Route', 'Additional_Info'], axis=1, inplace=True)

data.head(3)

data['Total_Stops'] = data['Total_Stops'].replace({'non-stop': 0, '2 stops': 2, '1 stop': 1, '3 stops': 3, '4 stops': 4})

data.head(3)

data.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)

# merge all dataframes in a single dataframe
train_data = pd.concat([data, airline_names, source, destination], axis=1)

train_data.head(4)

train_data.shape

test.head(3)

test['Date_of_Journey'] = pd.to_datetime(test['Date_of_Journey'], 
                                         infer_datetime_format=True)

test.drop(['Unnamed: 0', 'Route', 'Additional_Info'], axis=1, inplace=True)

test['Day of Journey'] = test['Date_of_Journey'].dt.day
test['Month of Journey'] = test['Date_of_Journey'].dt.month
test['Year of Journey'] = test['Date_of_Journey'].dt.year

# after making additional features from datetime, now we remove the datetime feature 
test.drop('Date_of_Journey', axis=1, inplace=True)

test['Dep_Time'] = pd.to_datetime(test['Dep_Time'], 
                                         infer_datetime_format=True)

test['Hour of Departure'] = test['Dep_Time'].dt.hour
test['Minute of Departure'] = test['Dep_Time'].dt.minute

# dropping original departure time feature
test.drop('Dep_Time', axis=1, inplace=True)

# dropping original departure time feature
# data.drop('Dep_Time', axis=1, inplace=True)

test['Arrival_Time'] = pd.to_datetime(test['Arrival_Time'], 
                                         infer_datetime_format=True)

test['Arrival Hour'] = test.Arrival_Time.dt.hour
test['Arrival Minute'] = test.Arrival_Time.dt.minute

test.drop('Arrival_Time', axis=1, inplace=True)

duration = list(test['Duration'])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + ' 0m'
        else:
            duration[i] = "0h " + duration[i]

duration_hrs = []
duration_mins = []

for i in range(len(duration)):
    duration_hrs.append(int(duration[i].split(sep='h')[0]))
    duration_mins.append(int(duration[i].split(sep='m')[0].split()[-1]))    

test['Duration Hours'] = duration_hrs
test['Duration Mins'] = duration_mins

airline_names = test[['Airline']]
airline_names = pd.get_dummies(airline_names, drop_first=True)

source_test = test[['Source']]
source_test = pd.get_dummies(source_test, drop_first=True)

dest_test = test[['Destination']]
dest_test = pd.get_dummies(dest_test, drop_first=True)

test.drop(['Airline', 'Source', 'Destination', 'Duration'], axis=1, inplace=True)

test.head(3)

# merge all dataframes in a single dataframe
test_data = pd.concat([test, airline_names, source_test, dest_test], axis=1)

test_data['Duration Hours']

test_data['Total_Stops'] = test_data['Total_Stops'].replace({'non-stop': 0, '2 stops': 2, '1 stop': 1, '3 stops': 3, '4 stops': 4})

test_data.shape

test_data.head(4)

X = train_data.drop('Price', axis=1)
Y = train_data['Price']

X.shape, Y.shape

X_train, X_val, Y_train, Y_val = train_test_split(X, 
                                                  Y, 
                                                  test_size=0.2, 
                                                  random_state=123)

print(X_train.shape)
print(X_val.shape)
print(Y_train.shape)
print(Y_val.shape)

import matplotlib.pyplot as plt

# plt.figure(figsize=(20, 20))
sns.set(rc={'figure.figsize':(25, 25)})
sns.heatmap(train_data.corr(),
           annot=True, cmap='RdYlGn')
plt.show()

%%time
rf = RandomForestRegressor()
rf.fit(X, Y)

rf.get_params()

plt.figure(figsize=(10, 5))
plt.title('Feature Importance')
imp = pd.Series(rf.feature_importances_, index=X.columns)
imp.nlargest(20).plot(kind='barh')
plt.show()

print('Accuracy of RF on train: %.4f' % rf.score(X_train, Y_train))
print('Accuracy of RF on validation: %.4f' % rf.score(X_val, Y_val))

y_pred = rf.predict(X_val)

sns.displot(Y_val-y_pred)

plt.figure(figsize=(8, 5))
plt.scatter(Y_val, y_pred, alpha=0.5, color='grey')

print('R2 Score of Model: %.4f' % r2_score(Y_val, y_pred))

import math

def show_eror():
    print('Mean Squared Error: %.3f' % MSE(Y_val, y_pred))
    print('Mean Absolute Error: %.3f' % MAE(Y_val, y_pred))
    print('Root Meab Squared Error: %.3f' % math.sqrt(MSE(Y_val, y_pred)))
    print('Root Meab Squared Error: %.3f' % math.sqrt(MAE(Y_val, y_pred)))

show_eror()

## Saving The Model

import pickle
from pickle import load

file = open('flight.pkl', 'wb')
pickle.dump(rf, file)

model = open('flight.pkl', 'rb')
rf_model = load(model)

rf_model

prediction = rf_model.predict(X_val)

prediction

Y_val_ = Y_val.to_numpy()

### Showing the Predictions Made by Model

df = pd.DataFrame({'Actual Price': Y_val, 'Predicted Price': y_pred})
df.head(10)

# plot model results
plt.figure(figsize=(15, 10))
plt.title('Random Forest Regression Results of Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.plot(Y_val_[:500], label='Actual Price')
plt.plot(y_pred[:500], label='Predicted Price')
plt.legend()

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameter={'alpha':[0.5,1,2,3,4,5,10,20,30,40,50]}
ridgecv=GridSearchCV(ridge,parameter,scoring='neg_mean_squared_error',cv=5)
ridgecv.fit(X_train,Y_train)

print(ridgecv.best_params_)
ridge_pred=ridgecv.predict(X_val)
sns.displot(ridge_pred,kind='kde')
sns.displot(Y_val,kind='kde')

from sklearn.metrics import  r2_score 
rscore=r2_score(ridge_pred,Y_val)
rscore

import sklearn.metrics as metrics
import numpy as np
ridge_prediction = ridgecv.predict(X_val)
mae_ridge = metrics.mean_absolute_error(Y_val, ridge_prediction)
mse_ridge =  metrics.mean_squared_error(Y_val, ridge_prediction)
rmse_ridge=  np.sqrt(mse_ridge)
print('MAE:', mae_ridge)
print('MSE:', mse_ridge)
print('RMSE:', rmse_ridge)
model=ridge.fit(X,Y)
print('Model coeff : ',model.coef_)
print('Model intercept : ',model.intercept_)
print('Accuracy : ', model.score(X,Y))

#Lasso Regreesion
from sklearn.linear_model import Lasso
lasso=Lasso()
parameter={'alpha':[0.01,0.001,0.0001,0.5,1]}
lassocv=GridSearchCV(lasso,parameter,scoring='neg_mean_squared_error',cv=5)
lassocv.fit(X_train,Y_train)

print(lassocv.best_params_)
lasso_pred=lassocv.predict(X_val)
lsscore=r2_score(lasso_pred,Y_val)
lsscore

import sklearn.metrics as metrics
lasso_prediction = lassocv.predict(X_val)
mae_lasso = metrics.mean_absolute_error(Y_val, lasso_prediction)
mse_lasso =  metrics.mean_squared_error(Y_val, lasso_prediction)
rmse_lasso=  np.sqrt(mse_lasso)
print('MAE:', mae_lasso)
print('MSE:', mse_lasso)
print('RMSE:', rmse_lasso)
model=lasso.fit(X,Y)
print('Model coeff : ',model.coef_)
print('Model intercept : ',model.intercept_)
print('Accuracy : ', model.score(X,Y))

#build and train a linear regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_val,Y_val)
plt.scatter(lr.predict(X_val), Y_val)
plt.xlabel('Predicted value of Y')
plt.ylabel('Real value of Y')
plt.show()

#linear regression
import sklearn.metrics as metrics
lr_prediction = lr.predict(X_val)
mae_lr = metrics.mean_absolute_error(Y_val, lr_prediction)
mse_lr =  metrics.mean_squared_error(Y_val, lr_prediction)
rmse_lr =  np.sqrt(mse_lr)
print('MAE:', mae_lr)
print('MSE:', mse_lr)
print('RMSE:', rmse_lr)
model=lr.fit(X,Y)
print('Model coeff : ',model.coef_)
print('Model intercept : ',model.intercept_)
print('Accuracy : ', model.score(X,Y))
