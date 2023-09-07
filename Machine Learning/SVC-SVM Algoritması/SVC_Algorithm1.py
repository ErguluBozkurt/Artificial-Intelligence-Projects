# Scikit-Learn'ün içine gömülü bulunan meme kanseri veri setini kullanacağız.

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 

cancer = load_breast_cancer()
print(cancer.keys())

df_feat = pd.DataFrame(cancer['data'],columns = cancer['feature_names'])
print("Öznitelikler: ")
print(df_feat.info())
print("Öznitelikleri içeren Veri çerçevemizi görelim : ")
df_feat.head() 

df_target = pd.DataFrame(cancer['target'],columns =['Cancer'])
df_target.tail()

X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target),test_size = 0.20) 

model = SVC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions)) # Sonuçları rapor şeklinde bize gösterecektir. 
print(confusion_matrix(y_test, predictions))

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
 
grid = GridSearchCV(SVC(), param_grid, refit = True)
 
grid.fit(X_train, y_train)

# Optimizasyondaki en iyi hiperparametre kombinasyonunu ekrana verelim.
print(grid.best_params_)

# En iyi sonucu veren hiperparametre kombinasyonunun doğruluğunu ekrana verelim.
print(grid.best_score_)
grid_predictions = grid.predict(X_test)
 
# Sınıflandırma raporunu ekrana verelim
print(classification_report(y_test, grid_predictions))
print(confusion_matrix(y_test, grid_predictions))