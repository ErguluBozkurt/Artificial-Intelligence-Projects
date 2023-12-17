import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

"""
Bu projede yetişkin bireylerin gelir seviyeleri incelenmiş ve K-Neighbors-Classifier algoritması
kullanılarak bireyler gelir verisine göre sınıflandırma işlemi yapılmıştır.

"""

data = pd.read_csv("adult.csv")
print(data.head())
print(data.info())
print(data.describe()) # age 18 altı gözüküyor incelecek
# Boş değer var mı?
print(data.isnull().sum()) # Boş değere sahip değiliz


#### Değişkenler Arası İlişki

for features_list in data.columns:
    features = data[features_list].unique()
    print("#"*25)
    print(f"{features_list.title()} Variables : {features} \nPiece:{len(features)}")
# Sonuç : Boş değerleri "?" ile doldurulmuş bunu kaldıralım

# age, workclass ve income inceleme
outliers = [i for i in data["age"] if i < 18]
print("#"*25)
print(f"18 den küçük kişi sayısı : {len(outliers)}")
# Sonuç : sadece 595 kişi 18 yaşından küçük

# 18 altı yaş gurubunu çıkartalım.
yas_sutunu = data["age"] >= 18
data = data[yas_sutunu]

outliers = [i for i in data["age"] if i < 18]
print("#"*25)
print(f"18 den küçük kişi sayısı : {len(outliers)}")

print(data.isin(['?']).sum()) 
# Sonuç : workclass, occupation ve native-country de var. Bunları nan değer ile değiştirelim

data['workclass']=data['workclass'].replace('?',np.nan)
data['occupation']=data['occupation'].replace('?',np.nan)
data['native-country']=data['native-country'].replace('?',np.nan)

print(data.head(10))

print(data.shape) # satır sayısı
# nan değerleri çıkaralım şimdi
data = data.dropna()
print(data.head(10))

print(data.shape) 
# Sonuç : satır sayısı düştü

info= pd.DataFrame(data.isnull().sum(),columns=["IsNull"])
info.insert(1,"Duplicate",data.duplicated().sum(),True)
info.insert(2,"Unique",data.nunique(),True)
info.insert(3,"Min",data.min(),True)
info.insert(4,"Max",data.max(),True)
print(info)
# Sonuç : 46 adet aynı satır var, çıkaralım
data = data.drop_duplicates() # satırlar çıkarıldı

print(data.shape) 
# Sonuç : satır sayısı düştü

# education ve educational-num aynı değerler olduğu için birini silelim. 
# capital-loss, capital-gain tekrar ediyor çıkaralım. 
data = data.drop(['educational-num', 'capital-loss', 'capital-gain'], axis=1) 
print(data.head())


# label encoder
label_encoder = LabelEncoder()
list_features = ['gender', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country', 'income'] # String sütunu seçin
for i in list_features:
    data[i] = label_encoder.fit_transform(data[i])

print(data.head())

# verileri çeşitli grafikler ile inceleyelim
def diagnostic_plots(df, variable, target):
    # histogram
    plt.figure(figsize=(20, 7))
    plt.subplot(1, 4, 1)
    sns.histplot(df[variable], kde=True, color='r')
    plt.title(f'{variable} Histogram')

    # scatterplot
    plt.subplot(1, 4, 2)
    plt.scatter(df[variable], df[target], color='g')
    plt.title(f'{variable} vs {target} Scatterplot')

    # boxplot
    plt.subplot(1, 4, 3)
    sns.boxplot(y=df[variable], color='b')
    plt.title(f'{variable} Boxplot')

    # barplot
    plt.subplot(1, 4, 4)
    sns.barplot(data=df, x=target, y=variable)
    plt.title(f'{variable} vs {target} Barplot ')

    plt.show()

for col in data:
    diagnostic_plots(data, col, 'income')





# Eğitim

X = data.drop("income", axis = 1).values
y = data["income"].values

# Veriyi eğitim ve test alt kümelerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

st =MinMaxScaler()
X_train = st.fit_transform(X_train)
X_test = st.fit_transform(X_test)

# En iyi komşu K değeri
K = 20
error =[]
accuracy=[]
for i in range(1,K+1):
    knn= KNeighborsClassifier(n_neighbors= i)
    knn.fit(X_train,y_train)
    y_pred =knn.predict(X_test)
    error.append(1-metrics.accuracy_score(y_test,y_pred)) # başarısız durum scoru
    accuracy.append(metrics.accuracy_score(y_test,y_pred)) # başarılı durum scoru

# Grafikte gösterelim
plt.figure(figsize=(20, 7))
plt.subplot(1, 2, 1)
plt.plot(range(1,21),error,'r-',marker='o')
plt.xlabel('K Değeri')
plt.ylabel('Hata')
plt.grid()
plt.title('Hata vs K')

plt.subplot(1, 2, 2)
plt.plot(range(1,21),accuracy,'r-',marker='o')
plt.xlabel('K Değeri')
plt.ylabel('Başarı')
plt.grid()
plt.title('Başarı vs K')
plt.show()



# Modelleri ve ilgili hiperparametre aralıklarını tanımlayın
models = {
        'name': 'KNeighborsClassifier',
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [16,17,18,19],
            'weights': ['uniform', 'distance'],
            "metric":["euclidean","manhattan"]
            }
        }


# GridSearchCV
model = GridSearchCV(models['model'], models['params'], cv=5, n_jobs=-1) 
model.fit(X_train, y_train) 

# En iyi sonuçlar
print(f"Model: {models['name']} \nBest Parameters: {model.best_params_} \nBest Score: {round(model.best_score_, 2)}%")
print("-" * 30)
    
predictions = model.predict(X_test)
print(classification_report(y_test, predictions)) 
print(confusion_matrix(y_test, predictions))

scorem = model.score(X_test, y_test) 
print(f"Test regrasyonun scoru : {round(scorem, 2)} ")

scorem = model.score(X_train, y_train) 
print(f"Eğitim regrasyonun scoru : {round(scorem, 2)} ")


# Test ve Train verisi arasında ilişki
training_acc = []
test_acc = []
neighbors_setting = range(1,21)
for n_neighbors in neighbors_setting:
    knn= KNeighborsClassifier(n_neighbors= n_neighbors)
    knn.fit(X_train,y_train.ravel())
    training_acc.append(knn.score(X_train,y_train))
    test_acc.append(knn.score(X_test,y_test))

plt.plot(neighbors_setting,training_acc,label='Eğitim Verisi')
plt.plot(neighbors_setting,test_acc,label='Test Verisi')
plt.ylabel('Başarı')
plt.xlabel('Komşu sayısı')
plt.grid()
plt.legend()
plt.show()