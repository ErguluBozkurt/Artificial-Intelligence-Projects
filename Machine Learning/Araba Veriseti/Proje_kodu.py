import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import  PolynomialFeatures  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold,  GridSearchCV
from sklearn.metrics import  r2_score
import operator
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# Değişkenlerin Açıklanması
"""
Car_Name : Araç Adı
Year : Aracın Satın Alındığı Yıl
* Selling_Price : Sahibinin Satmak İstediği Fiyat
Present_Price : Şuanki güncel fiyat
Kms_Driven : Kat Ettiği km Mesafesi
Fuel_Type : Yakıt Türü
Seller_Type : Satıcı bayi(dealer) mi yoksa şahıs(individual) mı?
Transmission : Araç manuel mi yoksa otomatik mi?
Owner : Aracın daha önce sahip olduğu sahip sayısı
"""

data = pd.read_csv("Machine Learning\Codes\Araba Veriseti\car data.csv")
print(data.head())
print(data.info())
print(data.describe())
# Boş değer var mı?
print(data.isnull().sum()) # Boş değere sahip değiliz 

sns.pairplot(data)  
plt.show() # Present_Price ile Selling_Price arasında bir ilişki var gibi duruyor



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
    
category = ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"] 
for i in category:
    bar_plot(i)
# Sonuç : city tipi araç yüksek satışa sahip, Yakıt türü yoğun olarak Petrol, bayi tipi satış ve manuel tipi araç satışı yoğunlukda
# Car_Name ayrıca incelenmesi gerekiyor
    
data_set = data["Car_Name"].value_counts().reset_index()
print(data_set.head(25))
sns.barplot(data_set[:25], x = "index", y = "Car_Name", palette = "coolwarm")
plt.xticks(rotation=80)
plt.show()



# Sayısal Verilerin Analizi

def plot_hist(variable):
    plt.figure(figsize=(7,3))
    plt.hist(data[variable], bins = 30)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Graphic of {variable}")
    plt.show()

numeric = ["Year", "Selling_Price", "Present_Price", "Kms_Driven", "Owner"] 
for i in numeric:
    plot_hist(i)
# Sonuç : 2015 yılında satışlarda artış var. Fiyatın artmasına göre satışlarda düşüş var.

corr_matrix = data.corr() 
print(corr_matrix["Selling_Price"].sort_values(ascending=False)) 
# Sonuç : Selling_Price, Present_Price ile yüksek korelasyona sahip



# GÖRSELLEŞTİRME
# Sayısal Analiz

list_value = ["Year", "Selling_Price", "Present_Price", "Kms_Driven", "Owner"] 
sns.heatmap(data[list_value].corr(), annot=True, fmt=".2f")
plt.show()


plt.figure(figsize=(12,6))
sns.barplot(data = data[::5], x = "Selling_Price", y = "Present_Price")
plt.xticks(rotation=80)
plt.show() 
# Sonuç : Doğrusal bir artış var

plt.figure(figsize=(12,6))
sns.barplot(data = data[::5], x = "Selling_Price", y = "Kms_Driven")
plt.xticks(rotation=80)
plt.show()
# Sonuç : Doğrusal bir artış yok

plt.figure(figsize=(12,6))
sns.barplot(data = data[::5], x = "Selling_Price", y = "Owner")
plt.xticks(rotation=80)
plt.show()
# Sonuç : Doğrusal bir artış yok



# Kategorik Analiz

fig = sns.FacetGrid(data, col = "Fuel_Type")
fig.map(sns.distplot, "Selling_Price")
plt.show()

fig = sns.FacetGrid(data, col = "Seller_Type")
fig.map(sns.distplot, "Selling_Price")
plt.show()

fig = sns.FacetGrid(data, col = "Transmission")
fig.map(sns.distplot, "Selling_Price")
plt.show()

fig = sns.FacetGrid(data, col = "Fuel_Type", row = "Seller_Type")
fig.map(sns.barplot, "Selling_Price", "Present_Price")
plt.show()
# Sonuç : Bireysel ve dizel araç fiyatı daha yüksek

df = data.copy()
sns.countplot(data = df, x = "Fuel_Type")
plt.show()
df = pd.get_dummies(df,columns=["Fuel_Type"]) # veriri kategorik değişkeni ikli sınıf değişkenlerine ayırır
print(df.head())

sns.countplot(data = df, x = "Seller_Type")
plt.show()
df = pd.get_dummies(df, columns=["Seller_Type"])

sns.countplot(data = df, x = "Transmission")
plt.show()
df = pd.get_dummies(df, columns= ["Transmission"])
print(df.head())

df = pd.get_dummies(df, columns=["Car_Name"])



# Aykırı Veri Analizi

list_features = ["Selling_Price", "Present_Price"] 
sns.boxplot(data = data.loc[:, list_features], orient = "v", palette = "Set1") 
plt.show()  
# Sonuç : Hem Selling_Price hemde Present_Price değişkeninde ayrık veri var

list_features = ["Year", "Kms_Driven", "Owner"] 
sns.boxplot(data = data.loc[:, list_features], orient = "v", palette = "Set1") 
plt.show()  
# Sonuç : Kms_Driven ayrık verilere sahip

# Ayrık verilerin indexlerini bul ve çıkar
def detect_outlier(df_, features):
    outlier_indices = []
    for i in features:
        q1 = np.percentile(df_[i], 25)
        q3 = np.percentile(df_[i], 75)
        iqr = q3 - q1
        outlier_step = iqr * 1.5
        outlier_list = df_[(df_[i] < q1 - outlier_step) | (df_[i] > q3 + outlier_step)].index
        outlier_indices.extend(outlier_list)

    return(outlier_indices)

outliers = detect_outlier(df, ["Selling_Price", "Present_Price", "Kms_Driven"])
print(outliers) # Aykırı değerler
df = df.drop(outliers, axis=0).reset_index(drop=True) # Aykırı değerleri çıkar
print(df.tail(10))

list_features = ["Selling_Price", "Present_Price", "Kms_Driven"] 
sns.boxplot(data = df.loc[:, list_features], orient = "v", palette = "Set1") 
plt.show()  



# Model Eğitimi

price_bins = [0, 7, 14, 21, 28, 35]  # Belirlediğiniz fiyat aralıkları
price_labels = ["Very Low", "Low", "Medium", "High", "Very High"]  # Her aralık için etiketler
# Veriyi sınıflandırma aralıklarına göre dönüştürün
df['Price_Category'] = pd.cut(df['Selling_Price'], bins=price_bins, labels=price_labels)

lenght = len(data)
print(lenght)
train = df[:lenght]
X_train = train.drop(labels = "Price_Category", axis = 1)
y_train = train["Price_Category"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)
print(f"X_train : {len(X_train)} \nX_test : {len(X_test)} \ny_train : {len(y_train)} \ny_test : {len(y_test)}")

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
    print(f"Model ve Parametriler : {best_estimators[i]} \nBaşarı Scoru : {round(cv_result[i], 2)}%")
    
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier","LogisticRegression","KNeighborsClassifier"]})

fig = sns.barplot(x = "Cross Validation Means", y = "ML Models", data = cv_results) # Başarı grafiği
plt.show()




# Polinomal Regresyon Uygulaması
df2 = data.copy()

# Ayrık verilerin indexlerini bul ve çıkar
def detect_outlier(df_, features):
    outlier_indices = []
    for i in features:
        q1 = np.percentile(df_[i], 25)
        q3 = np.percentile(df_[i], 75)
        iqr = q3 - q1
        outlier_step = iqr * 1.5
        outlier_list = df_[(df_[i] < q1 - outlier_step) | (df_[i] > q3 + outlier_step)].index
        outlier_indices.extend(outlier_list)

    return(outlier_indices)

outliers = detect_outlier(df2, ["Selling_Price", "Present_Price", "Kms_Driven"])
df2 = df2.drop(outliers, axis=0).reset_index(drop=True) # Aykırı değerleri çıkar

X = np.array(df2['Present_Price']).reshape(-1, 1) 
y = np.array(df2['Selling_Price']).reshape(-1, 1) #  reshape ile yeniden boyutlandırdık

X = np.log(X)
y = np.log(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

poli_reg = PolynomialFeatures(degree = 3) # polinom fonksiyonu tanımlanır.
transform_poli = poli_reg.fit_transform(X_train) # X eğitim verileri bu polinoma uydurulur ve dönüştürülür. 
dogrusal_reg2 = LinearRegression() # Şimdi, lineer regresyon fonksiyonumuzu çağırıyoruz. 
dogrusal_reg2.fit(transform_poli,y_train) 

transform_poli_test = poli_reg.fit_transform(X_test)
poli_tahmin_test = dogrusal_reg2.predict(transform_poli_test)
r2 = r2_score(y_test,poli_tahmin_test)
print("-"*30)
print("Polinomal Regresyon için R2 Skoru: " +"{:.2%}".format(r2))

plt.scatter(X_test, y_test) # Eğitim veri seti üzerine tahmini görselleştirelim.
sort_axis = operator.itemgetter(0) 
sorted_zip = sorted(zip(X_test,poli_tahmin_test), key=sort_axis)
X_test, poli_tahmin_test = zip(*sorted_zip)

plt.plot(X_test, poli_tahmin_test, color='r', label = 'Polinom Regresyon')
plt.xlabel('Present_Price') 
plt.ylabel('Selling_Price') 
plt.legend()
plt.show()




# Lineer Regresyon

X = np.array(df['Present_Price']).reshape(-1, 1)
y = np.array(df['Selling_Price']).reshape(-1, 1) #  reshape ile yeniden boyutlandırdık

X = np.log(X)
y = np.log(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
dogrusal_reg = LinearRegression() # Bu veriseti üzerinde Doğrusal Regresyon uygulayalım. 
dogrusal_reg.fit(X_train, y_train) 
y_pred = dogrusal_reg.predict(X_test)                                     
dogruluk_puani = dogrusal_reg.score(X_test, y_test)                       
print("Lineer Regresyon Modeli Dogruluk Puani: " + "{:.2%}".format(dogruluk_puani))

plt.scatter(X_test, y_test, color='r') # Doğrusal Regresyon modelini grafiğe dökelim.
plt.plot(X_test, y_pred, color='g')
plt.show()