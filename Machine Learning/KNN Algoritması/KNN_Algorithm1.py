"""
Bu projede beyin ağırlığını tespiti hedef alınmıştır. Yapılan çalışmalar ile en yüksek doğruluğun KNN algoritması 
kullanılarak bulunduğu gözlemlendi. 
Dataset:bas_beyin
"""

import pandas as pd
import matplotlib.pyplot as plt   
import seaborn as sns
from sklearn.model_selection import train_test_split
import operator
from sklearn.neighbors import KNeighborsRegressor


bas_beyin = pd.read_csv("bas_beyin.csv")
print(bas_beyin.head())

sns.pairplot(bas_beyin)  # Grafikleri oluşturuyor
plt.show()

plt.figure(figsize=(6, 6))
heatmap = sns.heatmap(bas_beyin.corr(), vmin=-1, vmax=1, annot = True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
plt.show() # Sıcaklık tablosunu gösterdi

X = bas_beyin['Bas_cevresi(cm^3)'].values  # Bağımsız Değişken 
y= bas_beyin['Beyin_agirligi(gr)'].values  # Bağımlı Değişken 

uzunluk = len(X)
X = X.reshape((uzunluk,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

model = KNeighborsRegressor(n_neighbors=1)
model.fit(X,y)

scorem = model.score(X_test, y_test) 
print(f"Regrasyonun scoru : {round(scorem, 2)} ")

sort_axis = operator.itemgetter(0) 
sorted_zip = sorted(zip(X_train,y_train), key=sort_axis)
X_train, y_train = zip(*sorted_zip)

plt.scatter(X_train, y_train, color = 'red') # Grafik çizdirildi
plt.plot(X_train, model.predict(X_train), color = 'blue')
plt.show()
