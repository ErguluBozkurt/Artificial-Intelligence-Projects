"""
Bu projede diyabet verisetini kullanarak diyabet kişlerin hastalığına sahip olup olmadığını sınıflandıran bir projedir. Başarı %80.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
"""
Pregnancies : Hamilelik sayısı
Glucose : Glikoz
BloodPressure : Kan basıncı
SkinThickness : Cilt kalınlığı
Insulin : İnsülin
BMI : Beden kitle endeksi
DiabetesPedigreeFunction : Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon
Age : Yaş
* Outcome : Kişinin diyabet olup olmadığı bilgisi.(hastalığa sahip 1, hasta değil 0)

"""

data = pd.read_csv("diabetes.csv") 

def check(df):
    print("  Shape  ".center(50, "#"))
    print(df.shape)
    print("\n")
    print("  Types  ".center(50, "#"))
    print(df.dtypes)
    print("\n")
    print("  Head  ".center(50, "#"))
    print(df.head())
    print("\n")
    print("  Tail  ".center(50, "#"))
    print(df.tail())
    print("\n")
    print("  Nan  ".center(50, "#"))
    print(df.isnull().sum())
    print("\n")
    print("  Quantiles  ".center(50, "#"))
    print(df.quantile([0, 0.05, 0.5, 0.95, 1]).T)

check(data)
# Sonuç : Verileri incelediğimizde sayısal ve float tipinde olduğunu görebiliyoruz. Aynı zamanda eksik veri içermiyor. Değişkenlerin içeriğini incelediğimizde
#         hamilelik sayısı 17 olması şüpheli bir durum ayrıca glikoz, kan basıncı, cilt kalınlığı insülin ve boy kitle endeksi min değerinin 0 çıkması yaşayan bir insan 
#         için mümkün olmadığı için incelenmesi lazım. Ayrıca insülin değerinin %99 da 519 olmasına rağmen max değeri 846 olması ani artış olduğu için ayrık değer şüphesi 
#         uyandırdı.
           

# Veri türlerini inceleyelim
def analyze_columns(df, threshold1=10, threshold2=20):
    categoric_cols = [col for col in df.columns if df[col].dtype == "O"]
    numeric_categoric = [col for col in df.columns if df[col].nunique() < threshold1 and df[col].dtype != "O"]
    categoric_cardinal = [col for col in df.columns if df[col].nunique() > threshold2 and df[col].dtype == "O"]
    categoric_cols = categoric_cols + numeric_categoric
    categoric_cols = [col for col in categoric_cols if col not in categoric_cardinal]

    numeric_cols = [col for col in df.columns if df[col].dtype != "O"]
    numeric_cols = [col for col in numeric_cols if col not in numeric_categoric]

    print(f"Toplam Satır : {df.shape[0]}")
    print(f"Toplam Sütun : {df.shape[1]}")
    print(f"Toplam Kategorik Sütun : {len(categoric_cols)}")
    print(f"Toplam Numeric Sutun : {len(numeric_cols)}")
    print(f"Toplam Kategorik Kardinal Sütun: {len(categoric_cardinal)}")
    print(f"Toplam Nümerik Kategorik Sütun: {len(numeric_categoric)}")
analyze_columns(data)

# Değişkenler ile hedef değişken arasındaki ilişki
for col in data.columns[:-1]:
    print(f" {col} - Outcome ".center(30, "#"))
    print(data[[col, "Outcome"]].groupby([col]).mean().sort_values(by="Outcome")[::-1].head())
    print(data[[col, "Outcome"]].groupby([col]).mean().sort_values(by="Outcome")[::-1].tail())
    print("-"*25)


# Korelasyon kontrolü
sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.show()
# Sonuç : Korelasyonlar düşük olduğu için sınıflandırma modeli bizim için daha uygun olacaktır

# Base model oluşturalım
y = data["Outcome"]
X = data.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Başarı oranı : {round(accuracy_score(y_pred, y_test), 2)}")
# Sonuç : Başarı %75

def plot_importance(model, features, num):
    feature_imp = pd.DataFrame({"Value" : model.feature_importances_, 
                                "Features" : features.columns})
    plt.figure(figsize=(10,10))
    sns.barplot(data=feature_imp.sort_values(by="Value", ascending=False)[0:num], x="Value", y="Features")
    plt.show()
plot_importance(model, X, len(X))
# Sonuç : Glikoz modelin başarısında yüksek oranda etkileyen değişken olmuş



##### EKSİK DEĞERLER ######
# Hamilelik ve Outcome değişkenleri dışındaki değişkenler 0 değerine sahip olamayacağı için bu değişkenleri Nan ile değiştirelim.
zero_columns = [col for col in data.columns if(data[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
print(f"Sıfır içeren sutunlat : {zero_columns}")

for col in zero_columns:
    data[col] = np.where(data[col]==0, np.nan, data[col]) 

# Boş değer sayısına bakalım
print(data.isnull().sum())

# Eksik verilerin oranı ne? Ne kadar kıymetli?
def missing_values(df):
    nan_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    num_miss = df[nan_columns].isnull().sum().sort_values(ascending=False)
    ratio = (df[nan_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([num_miss, np.round(ratio, 2)], axis=1, keys=["missing_number", "ratio"])
    print(missing_df, end="\n")
    print(f"\nTotal Missing Values : {num_miss.sum()}\n")
missing_values(data)  
# İnsülin eksik verileri, datanın %48 ini kaplıyor buda bizim için kıymetli 

# Boş değerlerin hedef değişken ile ilişkisi var mı?
for col in zero_columns:
    print(f" {col} - Outcome ".center(30, "#"))
    print(f"Eksik Değer Sayısı : {len(data.loc[data[col].isnull()])}")
    print(f"Dolu Değer Sayısı  : {len(data.loc[data[col].notnull()])}")
    mean_with_missing = data.loc[data[col].isnull(), "Outcome"].mean()
    print(f"Eksik Değer Oranı : {round(mean_with_missing, 2)}")
    mean_without_missing = data.loc[data[col].notnull(), "Outcome"].mean()
    print(f"Dolu Değer Oranı : {round(mean_without_missing, 2)}")
    print("-"*30)
# Sonuç : Cilt kalınlığı ve insüline ait boş değerler çok fazla hedef değişkeni etkilemiyor.
 
 # Eksik değerleri dolduralım
for col in zero_columns:
    data[col].fillna(data[col].median(), inplace=True) # medyan ile doldur

# Boş değer kontrolü yapalım
print(data.isnull().sum())



###### AYKIRI DEĞERLER ######
def outlier_tresholds(df, col_name, q1=0.05, q3=0.95):
    quartiel1 = df[col_name].quantile(q1)
    quartiel3 = df[col_name].quantile(q3)
    quartiel_range = quartiel3 - quartiel1
    up = quartiel3 + 1.5*quartiel_range
    down = quartiel3 - 1.5*quartiel_range
    return(up, down)

def check_outlier(df, col_name):
    up, down = outlier_tresholds(df, col_name)
    filter = df[(df[col_name] > up) | (df[col_name] < down)].any(axis=None) 
    if(filter == True):
        return(True)
    else:
        return(False)

def replace_outliers(df, col_name, q1=0.05, q3=0.95):
    up, down = outlier_tresholds(df, col_name)
    df.loc[(df[col_name] < down), col_name] = down
    df.loc[(df[col_name] > up), col_name] = up

for col in data.columns:
    print(f"{col} : {check_outlier(data, col)}") # Değişken içinde outlier var mı yok mu
    if(check_outlier(data, col)==True): # outlier varsa
        replace_outliers(data, col)

# Kontrol edelim
print("-"*30)
for col in data.columns:
    print(f"{col} : {check_outlier(data, col)}")



##### ÖZELLİK ÇIKARIMI #####
data["New_Age"] = [1 if(i >= 21 and i < 50) else 0 for i in data["Age"]]
data["New_BMI"] = ["underweight" if(i <= 18.5) else "healthy" if(i > 18.5 and  i <= 24.9) else "overweight" if(i > 24.9 and i <=29.9) else "obese" for i in data["BMI"]]
data["New_Glucose"] = ["normal" if(i <= 140) else "prediabetes" if(i > 140 and  i < 190) else "diabetes" for i in data["Glucose"]]
data["New_Age_BMI"] = [
    "underweight_mature" if (j <= 18.5 and (i >= 21 and i < 50)) else
    "underweight_senior" if (j <= 18.5 and i >= 50) else
    "healthy_mature" if ((j > 18.5 and j <= 24.9) and (i >= 21 and i < 50)) else
    "healthy_senior" if ((j > 18.5 and j <= 24.9) and i >= 50) else
    "overweight_mature" if ((j > 24.9 and j <= 29.9) and (i >= 21 and i < 50)) else
    "overweight_senior" if ((j > 24.9 and j <= 29.9) and i >= 50) else
    "obese_mature" if (j > 18.5 and (i >= 21 and i < 50)) else
    "obese_senior" for i, j in zip(data["Age"], data["BMI"])
]
data["New_Insulin"] = [1 if(i >= 16 and i <= 166) else 0 for i in data["Insulin"]]
data["Glıcose_Insulin"] = data["Glucose"] * data["Insulin"]
data["Glıcose_Pregnancies"] = data["Glucose"] * data["Pregnancies"]

print(data.head())



##### ONE-HOT-ENCODER ######
def one_hot_encoder(df, categorical_cols):
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return(df)
colums = [col for col in data.columns if(10 >= data[col].nunique() > 2)]

print(data[colums].head())
data = one_hot_encoder(data, colums)
print(data.head()) 



##### ÖZELLİK ÖLÇEKLENDİRME #####
def analyze_columns(df, threshold1=10, threshold2=20):
    categoric_cols = [col for col in df.columns if df[col].dtype == "O"]
    numeric_categoric = [col for col in df.columns if df[col].nunique() < threshold1 and df[col].dtype != "O"]
    categoric_cardinal = [col for col in df.columns if df[col].nunique() > threshold2 and df[col].dtype == "O"]
    categoric_cols = categoric_cols + numeric_categoric
    categoric_cols = [col for col in categoric_cols if col not in categoric_cardinal]

    numeric_cols = [col for col in df.columns if df[col].dtype != "O"]
    numeric_cols = [col for col in numeric_cols if col not in numeric_categoric]

    print(f"Toplam Satır : {df.shape[0]}")
    print(f"Toplam Sütun : {df.shape[1]}")
    print(f"Toplam Kategorik Sütun : {len(categoric_cols)}")
    print(f"Toplam Numeric Sutun : {len(numeric_cols)}")
    print(f"Toplam Kategorik Kardinal Sütun: {len(categoric_cardinal)}")
    print(f"Toplam Nümerik Kategorik Sütun: {len(numeric_categoric)}")
    return(numeric_cols)
numeric = analyze_columns(data)
scaler = StandardScaler()
data[numeric] = scaler.fit_transform(data[numeric])
print(data.head()) 




##### MODEL EĞİTİMİ ######
y = data["Outcome"]
X = data.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Başarı oranı : {round(accuracy_score(y_pred, y_test), 2)}")
# Sonuç : Başarı %79

def plot_importance(model, features, num):
    feature_imp = pd.DataFrame({"Value" : model.feature_importances_, 
                                "Features" : features.columns})
    plt.figure(figsize=(10,10))
    sns.barplot(data=feature_imp.sort_values(by="Value", ascending=False)[0:num], x="Value", y="Features")
    plt.show()
plot_importance(model, X, len(X))
# Sonuç : Glikoz*İnsilün modelin başarısında yüksek oranda etkileyen değişken olarak yerini aldı.
#         New_Age_Glucose başarıyı düşüren bir değişken olmuş durumda bu yüzden çıkarılıp değerlendirilmeli
