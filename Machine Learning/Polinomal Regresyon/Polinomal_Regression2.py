"""
Bu projede hedef sıcaklık değişkenini tespit etmektir.Sonuçda polinomal regresyon ile yüksek doğruluk elde edilmiştir.
Dataset:bottle
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures  
import operator



datas = pd.read_csv("bottle.csv")
print(datas.head())

datas_df = datas[['T_degC','R_SVA']] # 'T_degC' (Sicaklik),'R_SVA' sutunlarini ayiklayalim.
datas_df.columns = ['Sicaklik', 'Sva'] # Sutunları yeniden isimlendirelim yazması kolay olsun.

sns.pairplot(datas_df, kind="reg") # Verileri inceleyelim 
plt.show()

print(datas_df.isnull().sum()) # Null veri var mı onu kontrol edelim 
datas_df.fillna(method='ffill', inplace=True)
print(datas_df.isnull().sum() ) 

# Ayrık veriler çıkarıldı
min = datas_df['Sva']<80 
equal = datas_df['Sicaklik']>12
datas_df[min & equal] = np.nan
datas_df.fillna(method='ffill', inplace=True)

min = datas_df['Sva']<250 
max = datas_df['Sva']>230 
equal = datas_df['Sicaklik']>=11

print(datas_df[min & max & equal])
# y Bağımlı değişkenimiz, x bağımsız değişkenimiz. 
X = np.array(datas_df['Sva']).reshape(-1, 1)
y = np.array(datas_df['Sicaklik']).reshape(-1, 1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

poli_reg = PolynomialFeatures(degree = 4) 
transform_poli = poli_reg.fit_transform(X_train)  
dogrusal_reg2 = LinearRegression() 
dogrusal_reg2.fit(transform_poli,y_train) 

poli_tahmin = dogrusal_reg2.predict(transform_poli)
rmse = np.sqrt(mean_squared_error(y_train,poli_tahmin))
r2 = r2_score(y_train,poli_tahmin)
print("Test verisi için Kök Karesel Ortalama Hata: " +"{:.2}".format(rmse))
print("Test verisi için R2 Skoru: " +"{:.2}".format(r2))

plt.scatter(X_train, y_train) 
sort_axis = operator.itemgetter(0) 
sorted_zip = sorted(zip(X_train,poli_tahmin), key=sort_axis)
X_train, poli_tahmin = zip(*sorted_zip)
plt.plot(X_train, poli_tahmin, color='r', label = 'Polinom Regresyon')
plt.xlabel('Sva') 
plt.ylabel('Sıcaklık') 
plt.legend()
plt.show()
