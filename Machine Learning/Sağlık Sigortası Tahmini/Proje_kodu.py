# SVR Algoritması ile %84 başarı elde edilmiştir. 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR

# Değişkenlerin Açıklanması
"""
age : Yaş
sex : Cinsiyet (female, male)
bmi : Boy Kilo Endeksi (ideal: 18.5-24.9)
children : Sağlık sigortası kapsamındaki çocuk sayısı / Bakmakla yükümlü olunan kişi sayısı
smoker : Sigara içiyor mu? (yes, no)
region : Yerleşim alanı (northeast, southeast, southwest, northwest)
* charges : Sigorta masrafı

"""

data = pd.read_csv("Machine Learning\Codes\Sağlık Sigortası Tahmini\insurance.csv")
print(data.head())
print(data.info()) # sex, smoker ve region one hot encoder
print(data.describe())
# Boş değer var mı?
print(data.isnull().sum()) # Boş değere sahip değiliz

sns.pairplot(data)
plt.show() # Her hangi bir ilişki yok gibi duruyor



# Kategorik Veri Analizi

def bar_plot(variable):
    global n
    var = data[variable] # özellikleri al
    var_value = var.value_counts() # kategorik değişkenin adet sayısı

    plt.figure(figsize = (6,3)) # grafiği çizdir
    plt.bar(var_value.index, var_value)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()

category = ["sex", "smoker", "region"]
for i in category:
    bar_plot(i)
# Sonuç : sex ve region neredeyse eşit bir dağılıma sahipken smoker da sigara içen ile içmeyen arasında fark daha fazla

data_set = data["smoker"].value_counts().reset_index()
print(data_set.head(25))




# Sayısal Verilerin Analizi
sns.histplot(data=data, x = "charges", hue = "sex")
plt.show()
# Sonuç : Her hangi bir bilgi yok
sns.histplot(data=data, x = "charges", hue = "smoker")
plt.show()
# Sonuç : Sigara içmeyen kişlerin ücretleri düşük içenlerin ise yüksek gibi duruyor
sns.histplot(data=data, x = "charges", hue = "region")
plt.show()
# Sonuç : Her hangi bir bilgi yok

def plot_hist(variable):
    plt.figure(figsize=(7,3))
    plt.hist(data[variable], bins = 30)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Graphic of {variable}")
    plt.show()

numeric = ["age", "bmi", "children", "charges"]
for i in numeric:
    plot_hist(i)
# Sonuç : 18 yaşın kişi sayısı fazla. Endeksin çoğu 30. Çocucuksuz kişi sayısı fazla. Ücretler çoğunlukla düşük.

corr_matrix = data.corr()
print(corr_matrix["charges"].sort_values(ascending=False))
# Sonuç : Yüksek korelasyona sahip değiller




# DEĞİŞKENELR ARASI İLİŞKİ ANALİZİ
# Sayısal

# age - charges
print(data[["age", "charges"]].groupby(["age"]).mean().sort_values(by="charges")[::-1])
print("-"*25)
# Sonuç : Yaşı küçük olanlar daha az ücret ödüyor

# sex - charges
print(data[["sex", "charges"]].groupby(["sex"]).mean().sort_values(by="charges")[::-1])
print("-"*25)
# Sonuç : Kadın ve Erkek ücretlerinde ortalama aynı fiyatı ödüyor

# smoker - charges
print(data[["smoker", "charges"]].groupby(["smoker"]).mean().sort_values(by="charges")[::-1])
print("-"*25)
# Sonuç : Sigara içmeyenler daha az ücret ödüyor

# Region - charges
print(data[["region", "charges"]].groupby(["region"]).mean().sort_values(by="charges")[::-1])
print("-"*25)
# Sonuç : Bölgelerin ücrete herhangi bir etkisi yok




# GÖRSELLEŞTİRME
# Sayısal Analiz
list_value = ["charges", "age", "bmi", "children"]
sns.heatmap(data[list_value].corr(), annot=True, fmt=".2f")
plt.show()


plt.figure(figsize=(12,6))
sns.barplot(data = data[::20], x = "charges", y = "age")
plt.xticks(rotation=80)
plt.show()
# Sonuç : Az miktarda doğrusal bir artış var

plt.figure(figsize=(12,6))
sns.barplot(data = data[::20], x = "charges", y = "bmi")
plt.xticks(rotation=80)
plt.show()
# Sonuç : Doğrusal bir artış yok

plt.figure(figsize=(12,6))
sns.barplot(data = data[::20], x = "charges", y = "children")
plt.xticks(rotation=80)
plt.show()
# Sonuç : Doğrusal bir artış yok




# Kategorik Analiz
fig = sns.FacetGrid(data, col = "smoker")
fig.map(sns.distplot, "charges")
plt.show()

# Kategorik verileri dönüştürelim
df = data.copy()
string_sutunlar = ['smoker', 'region', 'sex']
def label_encode_column(df, column_name):
    label_encoder = LabelEncoder()
    df[column_name] = label_encoder.fit_transform(df[column_name])

for sutun in string_sutunlar:
    label_encode_column(df, sutun)
print(df.head())


# Aykırı Veri Analizi
list_features = ["age", "bmi", "children"]
sns.boxplot(data = data.loc[:, list_features], orient = "v", palette = "Set1")
plt.show()
# Sonuç : bmi, ayrık veriye sahip

sns.boxplot(data = data.loc[:, ["charges"]], orient = "v", palette = "Set1")
plt.show()
# Sonuç : charges ayrık verilere sahip

# Ayrık verilerin indexlerini bul ve ortalama ile değiştir
def replace_outliers_with_mean(df_, features):
    for i in features:
        q1 = np.percentile(df_[i], 30)
        q3 = np.percentile(df_[i], 55)
        iqr = q3 - q1
        outlier_step = iqr * 1.5
        lower_bound = q1 - outlier_step
        upper_bound = q3 + outlier_step
        outlier_indices = df_[(df_[i] < lower_bound) | (df_[i] > upper_bound)].index
        mean_value = df_[i].mean()
        df_.loc[outlier_indices, i] = mean_value
    return(df_)

df = replace_outliers_with_mean(df, ["bmi", "charges"])

sns.boxplot(data = df.loc[:, ["charges"]], orient = "v", palette = "Set1")
plt.show()




# Model Eğitilmesi
lenght = len(df)
train = df[:lenght]
X = train.drop(labels = "charges", axis = 1)
y = train["charges"]

min_value = 1e-10  # Eğer X veya y içinde sıfır veya negatif değerler varsa en küçük değeri belirle
X = np.log(X + min_value)
y = np.log(y + min_value)


# Veriyi eğitim ve test alt kümelerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Hiperparametrilerin seçilmesi
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']
              } 
 
regressor = GridSearchCV(SVR(), param_grid, refit = True)
regressor.fit(X_train, y_train)

# Optimizasyondaki en iyi hiperparametre kombinasyonunu
print(f"Best Parameters : {regressor.best_params_}")

scorem = regressor.score(X_test, y_test) 
print(f"Score : {round(scorem, 2)} ")

plt.scatter(y_train, regressor.predict(X_train), color="red", label="Eğitim Verileri")
plt.scatter(y_test, regressor.predict(X_test), color="blue", label="Test Verileri")
plt.title("SVR")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.legend()
plt.show()

