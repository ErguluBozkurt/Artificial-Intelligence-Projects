# Bu projede titanik gemi kazası veri seti incelenmiş ve makine öğrenmesi algoritmaları uygulanarak %84 başarı elde edilmiştir.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Değişkenlerin Açıklanması
"""
PassengerId : Yolcuların numara id sıralaması
Survived : Yolcular yaşıyor(1) ve yaşamıyor(0)
Pclass : Yolcuların sınıfı
Name : Yolcuların adı
Sex : Yolcuların cinsiyeti
Age : Yolcuların yaşı
SibSp : Yolcuların kardeşleri (siblings and spouses)
Parch : Ebebeyin ve çocukların sayısı (parents and children)
Ticket : Bilet numarası
Fare : Bilet fiyatı
Cabin : Kabin
Embarked : Yolcuların bindikleri limanlar
"""

train_data = pd.read_csv("train.csv")
print(train_data.head())
print(train_data.describe())
print(train_data.info())

test_data = pd.read_csv("test.csv")
print(test_data.head())

# Kategorik verilerin analizi
def bar_plot(variable):
    var = train_data[variable] # özellikleri al
    var_value = var.value_counts() # kategorik değişkenin adet sayısı
    
    plt.figure(figsize = (6,3)) # grafiği çizdir
    plt.bar(var_value.index, var_value)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()

category = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"] # grafiğini çizdirmek istediğimiz değişkenler
for i in category:
    bar_plot(i)

# Sayısal verilerin analizi
def plot_hist(variable):
    plt.figure(figsize=(7,3))
    plt.hist(train_data[variable], bins = 30)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Graphic of {variable}")
    plt.show()

numeric = ["Fare", "Age", "PassengerId"] # grafiğini çizdirmek istediğimiz değişkenler
for i in numeric:
    plot_hist(i)

# Pclass - Survived arasındaki ilişki
print(train_data[["Pclass", "Survived"]].groupby(["Pclass"]).mean().sort_values(by="Survived")[::-1])
print("-"*25)
# Sex - Survived arasındaki ilişki
print(train_data[["Sex", "Survived"]].groupby(["Sex"]).mean().sort_values(by="Survived")[::-1])
print("-"*25)
# SibSp - Survived arasındaki ilişki
print(train_data[["SibSp", "Survived"]].groupby(["SibSp"]).mean().sort_values(by="Survived")[::-1])
print("-"*25)
# Parch - Survived arasındaki ilişki
print(train_data[["Parch", "Survived"]].groupby(["Parch"]).mean().sort_values(by="Survived")[::-1])
print("-"*25)

# Aykırı değer tespiti
def detect_outlier(df, features):
    outlier_indices = []
    for i in features:
        q1 = np.percentile(df[i],25)
        q3 = np.percentile(df[i], 75)
        ıqr = q3 - q1
        outlier_step = ıqr * 1.5
        outlier_list = df[(df[i] < q1 - outlier_step) | (df[i] > q3 + outlier_step)].index
        outlier_indices.extend(outlier_list)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outlier = list(i for i, v in outlier_indices.items() if v > 2)
    return(multiple_outlier)
train_data.loc[detect_outlier(train_data, ["Age", "SibSp", "Parch", "Fare"])]

train_data = train_data.drop(detect_outlier(train_data, ["Age", "SibSp", "Parch", "Fare"]), axis=0).reset_index(drop = True) # Aykırı değerleri data dan çıkar
print(train_data.head(10))

# Kayıp verilerin tespiti
train_data_len = len(train_data)
train_data = pd.concat([train_data, test_data], axis = 0).reset_index(drop=True)
print(train_data.head())

print(train_data.columns[train_data.isnull().any()])
print(train_data.isnull().sum())
print(train_data[train_data["Embarked"].isnull()])

train_data.boxplot(column = "Fare", by = "Embarked")
plt.show()

train_data["Embarked"] = train_data["Embarked"].fillna("C") # Eksikverileri C limanı ile doldur
print(train_data[train_data["Embarked"].isnull()])

print(train_data[train_data["Fare"].isnull()])
train_data["Fare"] = train_data["Fare"].fillna(np.mean(train_data[train_data["Pclass"] == 3]["Fare"])) # Eksik veriyi doldur
print(train_data[train_data["Fare"].isnull()])


# Görselleştirme
list_value = ["SibSp", "Parch", "Age", "Fare", "Survived"] # SibSp, Parch, Age, Fare, Survived arasındaki korelasyon
sns.heatmap(train_data[list_value].corr(), annot=True, fmt=".2f")
plt.show()

sns.barplot(data = train_data, x = "SibSp", y = "Survived")
plt.show()

sns.barplot(data = train_data, x = "Parch", y = "Survived")
plt.show()

sns.barplot(data = train_data, x = "Pclass", y = "Survived")
plt.show()

fig = sns.FacetGrid(train_data, col = "Survived")
fig.map(sns.distplot, "Age", bins = 25)
plt.show()

fig = sns.FacetGrid(train_data, col = "Survived", row = "Pclass")
fig.map(plt.hist, "Age", bins = 25)
plt.show()

fig = sns.FacetGrid(train_data, row = "Embarked")
fig.map(sns.pointplot, "Pclass", "Survived", "Sex")
fig.add_legend()
plt.show()

fig = sns.FacetGrid(train_data, col = "Survived", row = "Embarked")
fig.map(sns.barplot, "Sex", "Fare")
plt.show()

print(train_data[train_data["Age"].isnull()]) # Yaşda bulunan eksik verilerin doldurulması
sns.boxplot(data = train_data, x="Sex", y="Age")
plt.grid(axis="y", linestyle='--')
plt.show() # sex, yaş ile ilgili bilgi içermiyor

sns.boxplot(data = train_data, x="Sex", y="Age", hue="Pclass")
plt.grid(axis="y", linestyle='--')
plt.show() # Pclass, yaş ile ilgili bilgi içeriyor

sns.boxplot(data = train_data, x="Parch", y="Age")
plt.grid(axis="y", linestyle='--')
plt.show() # Parch, yaş ile ilgili bilgi içeriyor

sns.boxplot(data = train_data, x="SibSp", y="Age")
plt.grid(axis="y", linestyle='--')
plt.show() # SibSp, yaş ile ilgili bilgi içeriyor

train_data["Sex"] = [1 if i=="male" else 0 for i in train_data["Sex"]]
sns.heatmap(train_data[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot=True)
plt.show()

# Tespit edilen değişkenlere bağlı olarak eksikler tamamlandı
index_nan_age = list(train_data["Age"][train_data["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_data["Age"][((train_data["SibSp"] == train_data.iloc[i]["SibSp"]) & (train_data["Parch"] == train_data.iloc[i]["Parch"]) & (train_data["Pclass"] == train_data.iloc[i]["Pclass"]))].median()
    age_med = train_data["Age"].median()
    if not np.isnan(age_pred):
        train_data["Age"].iloc[i] = age_pred
    else:
        train_data["Age"].iloc[i] = age_med
print(train_data[train_data["Age"].isnull()])


# Özellik mühendisliği ile yeni değişkenlerin tanımlanması
print(train_data["Name"].head())

name = train_data["Name"]
train_data["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
print(train_data["Title"].head())

sns.countplot( data = train_data, x="Title")
plt.xticks(rotation = 60)
plt.show()

# Kategorik hale getir
train_data["Title"] = train_data["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
train_data["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_data["Title"]]
print(train_data["Title"].head(20))

sns.countplot( data = train_data, x="Title")
plt.show()

fig = sns.barplot(data = train_data, x = "Title", y = "Survived")
fig.set_xticklabels(["Master","Mrs","Mr","Other"])
plt.show()

train_data.drop(labels = ["Name"], axis = 1, inplace = True)
print(train_data.head())

train_data = pd.get_dummies(train_data,columns=["Title"]) # veriri kategorik değişkeni ikli sınıf değişkenlerine ayırır
print(train_data.head())

train_data["Fsize"] = train_data["SibSp"] + train_data["Parch"] + 1
print(train_data.head())

fig = sns.barplot(data = train_data, x = "Fsize", y = "Survived")
plt.show()

train_data["family_size"] = [1 if i < 5 else 0 for i in train_data["Fsize"]]
print(train_data.head(10))

sns.countplot(data = train_data, x = "family_size")
plt.show()

fig = sns.barplot(data = train_data, x = "family_size", y = "Survived")
plt.show() # Small familes have more chance to survive than large families.

train_data = pd.get_dummies(train_data, columns= ["family_size"])
print(train_data.head())

print(train_data["Embarked"].head())

sns.countplot(data = train_data, x = "Embarked")
plt.show()

train_data = pd.get_dummies(train_data, columns=["Embarked"])
print(train_data.head())

print(train_data["Ticket"].head(10))

a = "A/5. 2151"
print(a.replace(".","").replace("/","").strip().split(" ")[0]) # örnek ayırma verisi

tickets = [] # tüm veriye uygulayalım
for i in list(train_data.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        tickets.append("x")
train_data["Ticket"] = tickets
print(train_data["Ticket"].head(20))

train_data = pd.get_dummies(train_data, columns= ["Ticket"], prefix = "T")
print(train_data.head(10))

sns.countplot(data = train_data, x = "Pclass")
plt.show()

train_data["Pclass"] = train_data["Pclass"].astype("category")
train_data = pd.get_dummies(train_data, columns= ["Pclass"])
print(train_data.head())

train_data["Sex"] = train_data["Sex"].astype("category")
train_data = pd.get_dummies(train_data, columns=["Sex"])
print(train_data.head())

train_data.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)
print(train_data.columns)

# Model eğitimi
lenght = train_data_len
print(lenght)

test = train_data[lenght:]
test.drop(labels = ["Survived"],axis = 1, inplace = True)
print(test.head())

train = train_data[:lenght]
X_train = train.drop(labels = "Survived", axis = 1)
y_train = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)
print(f"X_train : {len(X_train)} \nX_test : {len(X_test)} \ny_train : {len(y_train)} \ny_test : {len(y_test)} \ntest : {len(test)}")

# Hiperparametrilerin ve modelllerin seçilmesi
random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]
dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}
svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}
rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}
logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}
knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]

# Farklı Algoritmaların uygulanması. 
cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(X_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    
for i in range(len(classifier)):
    print("-"*30)
    print(f"Model ve Parametriler : {best_estimators[i]} \nBaşarı Scoru : {round(cv_result[i], 2)}")
    

cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

fig = sns.barplot(x = "Cross Validation Means", y = "ML Models", data = cv_results) # Başarı grafiği
plt.show()
