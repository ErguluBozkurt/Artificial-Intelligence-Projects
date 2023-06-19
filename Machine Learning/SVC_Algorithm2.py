"""
Projede kullanılan dataset iki çeşit kuru üzüm türünü içermektedir. Amaç girilen veriler doğrultusunda hangi türe
ait olduğunu bulmaktır. En iyi yöntemin SVC algoritmsı sonucuna ulaşılmıştır.
Dataset: Raisin_Dataset
"""

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


datas = pd.read_excel("Raisin_Dataset.xlsx")
print(datas.head())
print("#"*50)
print(datas.info())
print("#"*50)

# LabelEncoder'ı uygula
string_sutun = 'Class'
label_encoder = LabelEncoder()
datas[string_sutun] = label_encoder.fit_transform(datas[string_sutun])

print(datas["Class"])
# Korelasyon
corr_matrix = datas.corr()
print(corr_matrix["Class"].sort_values(ascending=False)) 

# Grafikleri oluştur
sns.pairplot(datas)  
plt.show()

# Yeni sütunu tanımla ve değerlerini belirle
datas['Type'] = (datas["MajorAxisLength"] - datas["MinorAxisLength"]) * datas["Perimeter"] * (datas["ConvexArea"] - datas["Area"])

X = np.array(datas["Type"]).reshape(-1, 1)
y = np.array(datas["Class"]).reshape(-1, 1)
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y),test_size = 0.20, random_state=42) 

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
 
regressor = GridSearchCV(SVC(), param_grid, refit = True)
 
# Kombinasyonları tek tek eğitelim
regressor.fit(X_train, y_train)

# Optimizasyondaki en iyi hiperparametre kombinasyonunu ekrana verelim.
print(regressor.best_params_)

scorem = regressor.score(X_test, y_test) # Sayısal hali
print(f"Score : {round(scorem, 2)} ")
grid_predictions = regressor.predict(X_test)
 
# Sınıflandırma raporunu ekrana verelim
print(classification_report(y_test, grid_predictions))
print(confusion_matrix(y_test, grid_predictions))




