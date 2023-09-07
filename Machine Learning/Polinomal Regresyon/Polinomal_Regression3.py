"""
Bu projede hedef sıcaklık değişkenini tespit etmektir.Sonuçda polinomal regresyon ile yüksek doğruluk elde edilmiştir.
Dataset:bottle
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
import operator

datas = pd.read_csv("bottle.csv")
print(datas.head())

datas_df = datas[['T_degC','STheta']] 

datas_df.columns = ['Sicaklik', 'heta'] # Sutunları yeniden isimlendirelim yazması kolay olsun.

sns.pairplot(datas_df, kind="reg") # Verileri inceleyelim 
plt.show()

print(datas_df.isnull().sum()) 

datas_df.fillna(method='ffill', inplace=True) # Null (NaN) verilerin olduğu satırları kaldıralım.
print(datas_df.isnull().sum() )  

x = np.array(datas_df['heta']).reshape(-1, 1) 
y = np.array(datas_df['Sicaklik']).reshape(-1, 1) 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=42)
poli_reg = PolynomialFeatures(degree = 2) 
transform_poli = poli_reg.fit_transform(X_train)  
dogrusal_reg = LinearRegression()
dogrusal_reg.fit(X_train,y_train) 

dogrusal_reg.fit(transform_poli,y_train)
poli_tahmin = dogrusal_reg.predict(transform_poli)

r2 = r2_score(poli_tahmin, y_train)
print("Polinom Regresyon Modeli Dogruluk Puani: " +"{:.2}".format(r2))

plt.scatter(X_train, y_train) 
sort_axis = operator.itemgetter(0) 
sorted_zip = sorted(zip(X_train,poli_tahmin), key=sort_axis)
X_train, poli_tahmin = zip(*sorted_zip)
plt.plot(X_train, poli_tahmin, color='r', label = 'Polinom Regresyon')
plt.xlabel('heta') 
plt.ylabel('Sıcaklık') 
plt.legend()
plt.show()
