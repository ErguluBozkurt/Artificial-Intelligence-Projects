"""
Bu projede hedef sıcaklık değişkenini tespit etmektir.Sonuçda lineer regresyon ile yüksek doğruluk elde edilmiştir.
Dataset:bottle
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 


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

dogrusal_reg = LinearRegression()
dogrusal_reg.fit(X_train,y_train) 
y_pred = dogrusal_reg.predict(X_test)  

polinom_egitim_dogruluk_puani = dogrusal_reg.score(X_test, y_test)                 
print("Lineer Regresyon için R2 Skoru: " + "{:.1%}".format(polinom_egitim_dogruluk_puani))

plt.scatter(X_train, y_train) 
plt.plot(X_test, y_pred, color='r', label = 'Lineer Regresyon')
plt.xlabel('heta') 
plt.ylabel('Sıcaklık') 
plt.legend()
plt.show()
