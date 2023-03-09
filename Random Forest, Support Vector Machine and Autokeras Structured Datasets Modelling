#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 23:18:30 2023

@author: engrimmanuel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearnex import patch_sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Activate scikit-learn-intelex to use Intel(R) Extension for Scikit-learn (previously known as DAAL)
patch_sklearn()

# Load data
df = pd.read_csv('/home/engrimmanuel/Desktop/jackfruit task image regression task /data/all-features-extraction-from-image-1-to-83-no-cracks.csv')
df = df.dropna()

# Define features and target variable
#X = df['equivalent_diameter'].values.reshape(-1, 1)
#X = df.loc[:,['equivalent_diameter','area','filled_area','feret_diameter_max']]
X = df.iloc[:, 1:15]
#X = X.drop('local_centroid',axis=1)
y = df['Gross weight (kg)'].values

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)


#############################################################################

# Define parameter grid for Random Forest
rf_param_grid = {'n_estimators': [100, 500, 1000],
                 'max_depth': [10, 20, 30, 40, None],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 4],
                 'max_features': ['auto', 'sqrt', 'log2']}

# Define Random Forest model and perform hyperparameter tuning
rf = RandomForestRegressor(n_jobs=-1)
rf_rs = RandomizedSearchCV(rf, rf_param_grid, n_iter=10, cv=5)
rf_rs.fit(X_train, y_train)

# Evaluate Random Forest model on validation set
rf_best=rf_rs.best_estimator_
y_val_pred_rf = rf_best.predict(X_val)
mse_rf = mean_squared_error(y_val, y_val_pred_rf)
r2=r2_score(y_val, y_val_pred_rf)

print('Random Forest RMSE on validation set:', mse_rf ** 0.5)
print('Random Forest R2_score on validation set:', r2)

#get the feature importances
importances = rf_best.feature_importances_
dataframe_importance =pd.DataFrame(importances).transpose()

indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])) 
    
from matplotlib import pyplot as plt

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()
####################################################################################################

import tensorflow as tf
import autokeras as ak

# Use GPU for training

# Initialize AutoKeras regressor
ak_reg = ak.StructuredDataRegressor(max_trials=100)

train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_set = tf.data.Dataset.from_tensor_slices((X_val, y_val))
x_val_ts = tf.data.Dataset.from_tensor_slices((X_val))
    
# Fit AutoKeras regressor to training data
ak_reg.fit(train_set, epochs=100)
    
# Evaluate AutoKeras regressor on validation set
mse_ak, r2_ak = ak_reg.evaluate(val_set)
    
# Make predictions on validation set
y_val_pred_ak = ak_reg.predict(x_val_ts)

mse_ann = mean_squared_error(y_val, y_val_pred_ak)
ann_R2_score = r2_score(y_val, y_val_pred_ak)
print('CUDA-based ANN RMSE on validation set:', mse_ann ** 0.5)
print ('ANN r2_score on validation set:', ann_R2_score)
    
# Print results
print('AutoKeras RMSE on validation set:', mse_ak ** 0.5)
print('AutoKeras R2_score on validation set:', r2_ak)
    
    
#################################################################################

# Define parameter grid for Support Vector Machine
svm_param_grid = {'C': [0.1, 1, 10, 100],
                  'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                  'gamma': ['scale', 'auto']}

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearnex import patch_sklearn
from sklearn.svm import SVR
patch_sklearn()

# Define CUDA-based Support Vector Machine model and perform hyperparameter tuning
cu_svm = SVR()
cu_svm_rs = RandomizedSearchCV(cu_svm, svm_param_grid, n_iter=10, cv=5)
cu_svm_rs.fit(X_train, y_train)

# Evaluate CUDA-based Support Vector Machine model on validation set
y_val_pred_cu_svm = cu_svm_rs.predict(X_val)
mse_cu_svm = mean_squared_error(y_val, y_val_pred_cu_svm)
mse_R2_score = r2_score(y_val, y_val_pred_cu_svm)
print('CUDA-based Support Vector Machine RMSE on validation set:', mse_cu_svm ** 0.5)
print ('Support Vector Machine r2_score on validation set:', mse_R2_score)

#################################################################################################################
