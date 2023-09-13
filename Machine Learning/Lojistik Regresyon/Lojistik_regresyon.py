"""
Bu projede kişide kalp olup olmadığı incelenmiştir. Lojistik regresyon ile en iyi tahminleme seviyesine ulaşılmıştır.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Değişkenlerin Açıklanması
"""
Age:     Age of the patient
Sex:     Sex of the patient
exang:   exercise induced angina (1 = yes; 0 = no)
ca:      number of major vessels (0-3)
cp:      Chest Pain type chest pain type
Value 1: typical angina
Value 2: atypical angina
Value 3: non-anginal pain
Value 4: asymptomatic
trtbps:  resting blood pressure (in mm Hg)
chol:    cholestoral in mg/dl fetched via BMI sensor
fbs:     (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
rest_ecg: resting electrocardiographic results
Value 0: normal
Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
thalach: maximum heart rate achieved
target:  0 = less chance of heart attack, 1 = more chance of heart attack
"""

data = pd.read_csv("Try\heart.csv")
print(data.head())
print(data.describe())
print(data.info())
print(data.columns)

# Boş değer var mı?
print(data.isnull().sum())

# Değişken analizi
print(data["sex"].value_counts())
for i in list(data.columns):
    print(f"{i} : {data[i].value_counts().shape[0]}")
    
# Karegorik veri analizi
categorical_list = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall", "output"]
data_categorical = data.loc[:, categorical_list]
for i in categorical_list:
        sns.countplot(data = data_categorical, x = i, hue="output")
        plt.title(i)
        plt.show()   
    
    
# Sayısal veri analizi
numeric_list = ["age", "trtbps", "chol", "thalachh", "oldpeak", "output"]
data_numeric = data.loc[:, numeric_list]
sns.pairplot(data = data_numeric, hue="output", diag_kind = "kde")
plt.show()

# Standardizasyon
scaler = StandardScaler()
scaled_array = scaler.fit_transform(data[numeric_list[:-1]]) # output dışında 
data_dummy = pd.DataFrame(scaled_array, columns=numeric_list[:-1])
print(data_dummy.head())
    
data_dummy = pd.concat([data_dummy, data.loc[:, "output"]], axis=1) # verisetini birleştir
print(data_dummy.head())
    
print(pd.DataFrame(scaled_array).describe())
    
# Görselleştirme
# boxplot
data_melted = pd.melt(data_dummy, id_vars="output", var_name="features", value_name="value")
sns.boxplot(data = data_melted, x="features", y="value", hue="output")
plt.show()

# swarm plot
sns.swarmplot(data = data_melted, x="features", y="value", hue="output")
plt.show()

# cat plot
sns.catplot(data = data, x="exng", y="age", hue="output", col="sex", kind="swarm")
plt.show()

sns.catplot(data = data, x="thall", y="age", hue="output", col="sex", kind="swarm")
plt.show()

sns.catplot(data = data, x="cp", y="age", hue="output", col="sex", kind="swarm")
plt.show()   
    
# Korelasyon
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.show()
    
# Ayrık veri analizi    
numeric_list = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
data_numeric = data.loc[:, numeric_list]
print(data_numeric.head())
    
# Ayrık verileri çıkart
for i in numeric_list:
    # IQR
    q1 = np.percentile(data.loc[:, i], 25) 
    q3 = np.percentile(data.loc[:, i], 75)
    IQR = q3 - q1
    print(f"Old Shape : {data.loc[:, i].shape[0]}")
    
    # üst sınır
    upper = np.where(data.loc[:,i] >= (q3 + 2.5 * IQR))
    # alt sınır
    lower = np.where(data.loc[:,i] <= (q1 - 2.5 * IQR))
    
    try:
        data.drop(upper[0], inplace=True)
    except:
        print("KeyError not found in axis")
    try:
        data.drop(lower[0], inplace=True)
    except:
        print("KeyError not found in axis")
    print(f"New Shape : {data.loc[:, i].shape[0]}\n")
    

# one-hot encoder
df = data.copy() 
df = pd.get_dummies(df, columns=categorical_list[:-1], drop_first=True)
print(df.head())

# Model
X = df.drop(["output"], axis=1)
y = df[["output"]]

scaler = StandardScaler()
X[numeric_list[:-1]] = scaler.fit_transform(X[numeric_list[:-1]]) 
print(X.head())

# Lojistik regresyon
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
reg = LogisticRegression()
reg.fit(X_train, y_train)
y_pred_prob = reg.predict_proba(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
dummy = pd.DataFrame(y_pred_prob)
dummy["y_pred"] = y_pred
dummy["y_test"] = y_test.reset_index(drop=True)
print(dummy.head(10))

print(f"Test accuracy : {round(accuracy_score(y_pred, y_test), 2)}")
