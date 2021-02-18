import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

# Importing dataset and examining it
dataset = pd.read_csv("PricePrediction.csv")
pd.set_option('display.max_columns', None) # to make sure you can see all the columns in output window
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Converting Categorical features into Numerical features
def converter(column):
    if column == 'Y':
        return 1
    else:
        return 0

dataset['CentralAir'] = dataset['CentralAir'].apply(converter)
dataset['PavedDrive'] = dataset['PavedDrive'].apply(converter)
print(dataset.info())

# Dividing dataset into label and feature sets
X = dataset.drop('SalePrice', axis = 1) # Features
Y = dataset['SalePrice'] # Labels
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

print(X_train.shape)
print(X_test.shape)

# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
rfr = RandomForestRegressor(criterion='mse', max_features='sqrt', random_state=1)
grid_param = {'n_estimators': [50,100,150,200,250]}

gd_sr = GridSearchCV(estimator=rfr, param_grid=grid_param, scoring='r2', cv=5)

gd_sr.fit(X_train, Y_train)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

# Building random forest using the tuned parameter
rfr = RandomForestRegressor(n_estimators=100, criterion='mse', max_features='sqrt', random_state=1)
rfr.fit(X_train,Y_train)
featimp = pd.Series(rfr.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimp)

Y_pred = rfr.predict(X_test)
print('MSE: ', metrics.mean_squared_error(Y_test, Y_pred))
print('R2 score: ', metrics.r2_score(Y_test, Y_pred))

# Selecting features with higher sifnificance and redefining feature set
X = dataset[['GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'GarageArea']]

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

rfr = RandomForestRegressor(n_estimators=100, criterion='mse', max_features='sqrt', random_state=1)
rfr.fit(X_train,Y_train)

Y_pred = rfr.predict(X_test)
print('MSE: ', metrics.mean_squared_error(Y_test, Y_pred))
print('R2 score: ', metrics.r2_score(Y_test, Y_pred))
