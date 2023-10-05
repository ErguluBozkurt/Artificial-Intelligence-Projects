import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.svm import SVC
import plotly.express as px 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import missingno as msno
from sklearn import tree


data = pd.read_csv("water_potability.csv")
print(data.head())
print(data.info()) 
print(data.describe())

sns.pairplot(data)
plt.show() # Her hangi bir ilişki yok gibi duruyor

d = pd.DataFrame(data["Potability"].value_counts())
fig = px.pie(d, values="Potability", names=["Not Potable", "Potable"], hole=0.35, opacity=0.9,
             labels={"label": "Potability", "count": "Number of Samples"})
fig.update_layout(title = dict(text = "Pie Chart of Potability Feature"))
fig.update_traces(textinfo = "percent+label")
fig.show()

sns.clustermap(data.corr(), cmap="vlag", dendrogram_ratio=(0.1,0.2), annot = True, figsize=(10,8))
plt.show() 
# sonuç : yüksek korelasyon yok

non_potable = data.query("Potability == 0")
potable = data.query("Potability == 1")

plt.figure(figsize = (15,15))
for ax, col in enumerate(data.columns[:9]):
    plt.subplot(3,3, ax + 1)
    plt.title(col)
    sns.kdeplot(x = non_potable[col], label = "Non Potable")
    sns.kdeplot(x = potable[col], label = "Potable")
    plt.legend()
plt.show() 
# sonuç :  değişkenler arası ilişki yok

msno.matrix(data)
plt.show()
# sonuç : ph, sulfate and Trihalomethanes kayıp değere sahip

# Boş değer var mı?
print(data.isnull().sum())
# kayıp değerler ortala ile dolduruldu
data["ph"].fillna(value = data["ph"].mean(), inplace=True)
data["Sulfate"].fillna(value = data["Sulfate"].mean(), inplace=True)
data["Trihalomethanes"].fillna(value = data["Trihalomethanes"].mean(), inplace=True)
print(data.isnull().sum())

# Model Rğitimi
X = data.drop("Potability", axis = 1).values
y = data["Potability"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("X_train",X_train.shape)
print("X_test",X_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)

# min-max Normalizasyon
x_train_max = np.max(X_train)
x_train_min = np.min(X_train)
X_train = (X_train - x_train_min)/(x_train_max-x_train_min)
X_test = (X_test - x_train_min)/(x_train_max-x_train_min)

models = [("DTC", DecisionTreeClassifier(max_depth = 3)),
          ("RF",RandomForestClassifier())]

finalResults = []
cmList = []
for name, model in models:
    model.fit(X_train, y_train) # train
    model_result = model.predict(X_test) # prediction
    score = precision_score(y_test, model_result)
    cm = confusion_matrix(y_test, model_result)
    
    finalResults.append((name, score))
    cmList.append((name, cm))

for name, i in cmList:
    sns.heatmap(i, annot = True, linewidths = 0.8, fmt = ".2f")
    plt.title(name)
    plt.show()
    
dt_clf = models[0][1]
plt.figure(figsize = (10,8))
tree.plot_tree(dt_clf,
               feature_names =  data.columns.tolist()[:-1],
               class_names = ["0", "1"],
               filled = True,
               precision = 5)
plt.show()

model_params = {
    "Random Forest":
    {
        "model":RandomForestClassifier(),
        "params":
        {
            "n_estimators":[10, 50, 100],
            "max_features":["auto","sqrt","log2"],
            "max_depth":list(range(1,15,3))
        }
    },
    "SVC":
    {
        'model': SVC(),
        'params': 
        {
            "gamma": [0.001, 0.01, 0.1, 1],
            'C': [1,10,50,100,200,300,1000],
            'kernel': ['rbf'],
        }
    },
    "DecisionTreeClassifier":
    {
        'model': DecisionTreeClassifier(),
        'params': 
        {
            'criterion': ['gini','entropy'],
            "max_depth": range(1,20,2),
        }
    },
    "KNeighborsClassifier":
    {
        'model': KNeighborsClassifier(),
        'params': 
        {
            'n_neighbors': [2,3,4,5,6,7,9,11,13,15,17,19],
            'weights': ['uniform', 'distance'],
            "metric":["euclidean","manhattan"]
        }
    }
}

cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 2)
scores = []
for model_name, params in model_params.items():
    rs = RandomizedSearchCV(params["model"], params["params"], cv = cv, n_iter = 10)
    rs.fit(X,y)
    scores.append([model_name, dict(rs.best_params_),rs.best_score_])
    
for i in scores:
    print("#"*15)
    print(f"Model : {i[0]} \nParametreler : {i[1]} \nBaşarı Scoru : {round(float(i[2]), 2)}")
    
    