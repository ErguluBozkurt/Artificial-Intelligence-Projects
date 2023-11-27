import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


"""
age                             : Yaş
gender                          : Cinsiyet (sex)
school ID                       : Okul numarası
size of family                  : Ailedeki kişi sayısı (famsize)
Father education                : Baba eğitimi (Fedu)
Mother education                : Anne eğitimi (Medu)
Occupation of Father and Mother : Baba ve Anne'nin mesleği (Fjob, Mjob)
Family Relation                 : Aile İlişkisi
Health                          : Sağlık
* Grades                        : Başarı seviyesi(G1, G2, *G3)

Bu projede öğrenci performansı verisini kullanarak regression örneği yapacaz.
Bu veriler ortaokuldaki öğrencilerin matematik derslerine ilişkin bir anketten elde edilmiştir.
"""

data = pd.read_csv('student_data.csv') 
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

##### Kategorik Analiz
# pie
categorical_columns = data.select_dtypes(include=['object']).columns
lenght = len(categorical_columns)
print(lenght)
plt.figure(figsize=(20, 15))

for i, col in enumerate(categorical_columns, start=1):
    plt.subplot(5, 4, i)
    data[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'{col} Kategorik Değişkeninin Pasta Grafiği')

plt.tight_layout()
plt.show()


##### Sayısal Analiz
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
 
for col in data.columns[-3:-1]:
    print(col)
    diagnostic_plots(data, col, 'G3')


#### Değişkenler Arası İlişki
for features_list in data.columns:
    features = data[features_list].unique()
    print("#"*25)
    print(f"{features_list.title()} Variables : {features} \nPiece:{len(features)}")


for i in data.columns[:-1]:
    print(data[[i, "G3"]].groupby([i]).mean().sort_values(by="G3")[::-1])
    print("-"*25)


sns.lmplot(x='G1',y='G3',data=data) 
plt.show()

sns.lmplot(x='G2',y='G3',data=data)
plt.show()


corr_matrix = data.corr() 
print(corr_matrix["G3"].sort_values(ascending=False)[:5]) 
# Sonuç : G1,G2 ve G3 arasında yüksek korelasyon var

list_value = ["G1", "G2", "G3"]
sns.heatmap(data[list_value].corr(), annot=True, fmt=".2f")  
plt.show() 

plt.figure(figsize=(25, 15))
plt.subplots_adjust(wspace=0.5, hspace=1.5)
for i,sutun in enumerate(data.columns[:-4]):
    plt.subplot(6,5,i+1)
    data[sutun].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title(f'{sutun} Değişkeninin Dağılımı')
    plt.xlabel(sutun)
    plt.ylabel('Frekans')
plt.show()

# label encoder
label_encoder = LabelEncoder()
list_features = data.select_dtypes(include=['object']).columns 
for i in list_features:
    data[i] = label_encoder.fit_transform(data[i])

print(data.head())



    

# Eğitim
X = data.drop("G3", axis = 1).values
y = data["G3"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc = StandardScaler()
X = sc.fit_transform(X_train)
y = sc.fit_transform(y_test.reshape(-1,1))

# Modelleri ve ilgili hiperparametre aralıklarını tanımlayın
models = [
    {
        'name': 'DecisionTreeRegressor',
        'model': DecisionTreeRegressor(),
        'params': {
            'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
            "max_depth": range(1,20,2),
        }
    },
    {
        'name': 'SVR',
        'model': SVR(),
        'params': {
            "gamma": [0.001, 0.01, 0.1, 1],
            'C': [1,10,50,100,200,300,1000],
            'kernel': ['rbf'],
        }
    },
    {
        'name': 'RandomForestRegressor',
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [100, 200, 300, 400],
            'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
        }
    },
    {
        'name': 'KNeighborsRegressor',
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [2,3,5,7,9],
            'weights': ['uniform', 'distance'],
        }
    },

]


best_models = []
for model_info in models:
    grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, n_jobs=-1) 
    grid_search.fit(X_train, y_train) 

    best_models.append({
        'name': model_info['name'],
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    })

# En iyi sonuçlar
for best_model in best_models:
    print(f"Model: {best_model['name']} \nBest Parameters: {best_model['best_params']} \nBest Score: {round(best_model['best_score'], 2)}%")
    print("-" * 30)

model_names = []
model_scores = []
for best_model in best_models:
    score = best_model['best_score']
    if score >= 0:
        model_names.append(best_model['name'])
        model_scores.append(score)

plt.figure(figsize=(12, 6))
plt.bar(model_names, model_scores, color='skyblue')
plt.xlabel('Model Score')
plt.title('Model Performansı')
plt.show()


