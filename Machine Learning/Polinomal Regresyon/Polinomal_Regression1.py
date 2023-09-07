"""
Bu projede hedef sıcaklık değişkenini tespit etmektir.Sonuçda polinomal regresyon ile yüksek doğruluk elde edilmiştir.
Dataset:bottle
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
import operator


datas = pd.read_csv("bottle.csv")
print(datas.head())

datas_df = datas[['T_degC','Salnty']] # 'T_degC' (Sicaklik),'Salnty' (Tuzluluk) sutunlarini ayiklayalim.
datas_df.columns = ['Sicaklik', 'Tuzluluk'] # Sutunları yeniden isimlendirelim yazması kolay olsun.

sns.pairplot(datas_df, kind="reg") # Verileri inceleyelim 
plt.show()

print(datas_df.isnull().sum()) # Null veri var mı 

datas_df.fillna(method='ffill', inplace=True) # Null (NaN) verilerin olduğu satırları kaldıralım.
print(datas_df.isnull().sum() )

# boş değerlerin adreslerini bulma 
for satir_numarasi, satir in enumerate(datas):  
    for sutun_numarasi, hucre in enumerate(satir):
        if not hucre.strip():
            print('Boş hücre bulundu:', satir_numarasi+1, sutun_numarasi+1)


# y Bağımlı değişkenimiz, x bağımsız değişkenimiz.
X = np.array(datas_df['Tuzluluk']).reshape(-1, 1) 
y = np.array(datas_df['Sicaklik']).reshape(-1, 1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

poli_reg = PolynomialFeatures(degree = 4) 
transform_poli = poli_reg.fit_transform(X_train) 
dogrusal_reg2 = LinearRegression()
dogrusal_reg2.fit(transform_poli,y_train) 

poli_tahmin = dogrusal_reg2.predict(transform_poli) 

rmse = np.sqrt(mean_squared_error(y_train,poli_tahmin))
r2 = r2_score(y_train,poli_tahmin)
print("Polinomal Ortalama Hata: " +"{:.2}".format(rmse))
print("Polinomal Skoru: " +"{:.2}".format(r2))

plt.scatter(X_train, y_train) # Eğitim veri seti üzerine tahmini görselleştirelim.
sort_axis = operator.itemgetter(0) 
sorted_zip = sorted(zip(X_train,poli_tahmin), key=sort_axis)
X_train, poli_tahmin = zip(*sorted_zip)
plt.plot(X_train, poli_tahmin, color='r', label = 'Polinom Regresyon')
plt.xlabel('Tuzluluk') 
plt.ylabel('Sıcaklık') 
plt.legend()
plt.show()

# Polinomal Regresyon sonuçlarına göre 33.82 tuzluluk derecesine karşılık gelen bir örneğin sıcaklığını tahmin etmeye çalışalım.
print(dogrusal_reg2.predict(poli_reg.fit_transform([[33.82]])))