"""
Bu projede twitter kullanıcıları tarafından atılan tweetlerin taşıdıkları duygu kapsamında
pozitif, negatif ve nötr olarak tahmin edilmesi amaçlanmıştır.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer



def data_preparation(dataframe, tf_idfVectorizer):
    
    # date değişkeninin zaman diliminin İstanbul zaman dilimine çevrilmesi
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe['date'] = dataframe['date'].dt.tz_convert('Europe/Istanbul')
    dataframe['date'] = dataframe['date'].dt.tz_localize(None)


    #### Feature Engineering
    # "month" değişkeninn oluşturulması ve düzenlenmesi
    dataframe['month'] = dataframe['date'].dt.month_name()
    dataframe["month"] = dataframe['month'].replace({ 'December': 'Aralık',
                                            'January': 'Ocak',
                                            'February': 'Şubat',
                                            'March': 'Mart',
                                            'April': 'Nisan',
                                            'May': 'Mayıs',
                                            'June': 'Haziran',
                                            'July': 'Temmuz',
                                            'August': 'Ağustos',
                                            'September': 'Eylül',
                                            'October': 'Ekim',
                                            'November': 'Kasım'
                                            })

    # "seasons" değişkeninin oluşturlması
    seasons = {'Ocak': 'Kış',
            'Şubat': 'Kış',
            'Mart': 'İlkbahar',
            'Nisan': 'İlkbahar',
            'Mayıs': 'İlkbahar',
            'Haziran': 'Yaz',
            'Temmuz': 'Yaz',
            'Ağustos': 'Yaz',
            'Eylül': 'Sonbahar',
            'Ekim': 'Sonbahar',
            'Kasım': 'Sonbahar',
            'Aralık': 'Kış'}

    dataframe['seasons'] = dataframe['month'].map(seasons)

    # gün değişkeninin oluşturulması
    dataframe["days"] = [date.strftime('%A') for date in dataframe["date"]]
    dataframe["days"] = dataframe["days"].replace({"Monday" : "Pazartesi",
                                        "Tuesday" : "Salı",
                                        "Wednesday" : "Çarşamba",
                                        "Thursday": "Perşembe",
                                        "Friday" : "Cuma",
                                        "Saturday" : "Cumartesi",
                                        "Sunday": "Pazar"})

    # 4 saatlik aralıklarla günün altıya bölünmesi
    dataframe['hour'] = dataframe['date'].dt.hour
    dataframe['4hour_interval'] = (dataframe['hour'] // 2) * 2
    interval = {0: '0-2',
                2: '2-4',
                4: '4-6',
                6: '6-8',
                8: '8-10',
                10: '10-12',
                12: '12-14',
                14: '14-16',
                16: '16-18',
                18: '18-20',
                20: '20-22',
                22: '22-24'
                }
    dataframe['4hour_interval'] = dataframe['4hour_interval'].map(interval)
    dataframe["time_interval"] = dataframe["4hour_interval"].replace({"0-2": "22-02",
                                                    "22-24": "22-02",
                                                    "2-4": "02-06",
                                                    "4-6": "02-06",
                                                    "6-8": "06-10",
                                                    "8-10": "06-10",
                                                    "10-12": "10-14",
                                                    "12-14": "10-14",
                                                    "14-16": "14-18",
                                                    "16-18": "14-18",
                                                    "18-20": "18-22",
                                                    "20-22": "18-22"})

    # Bağımlı değişkeni yeniden isimlendir
    dataframe["label"].replace(1, value="pozitif", inplace=True)
    dataframe["label"].replace(-1, value="negatif", inplace=True)
    dataframe["label"].replace(0, value="nötr", inplace=True)

    # tweetlerin küçük harfe çevrilemsi
    dataframe['tweet'] = dataframe['tweet'].str.lower() 

    # Gereksiz verileri çıkar
    dataframe.drop(["hour", "tweet_id", "date", "month"], axis=1, inplace=True)

    print(dataframe.head())
    print(dataframe.info())


    # Label Encoder
    encoder = LabelEncoder()
    dataframe["label"] = encoder.fit_transform(dataframe["label"]) # 0 = negatif, 1 = nötr, 2 = pozitif

    dataframe.dropna(axis=0, inplace=True)
    X = dataframe["tweet"]
    y = dataframe["label"] # Bağımlı değişken
    
    # Count Vectors Yöntemi (Frekans Temsiller)
    X = tf_idfVectorizer.fit_transform(X)

    return(X, y)


# Logistic Regression modelinin kurulması
def logistic_regression(X, y):
    model = LogisticRegression(max_iter=10000, n_jobs=-1)
    model.fit(X, y)
    print("Score : ", round(cross_val_score(model, X, y, scoring="accuracy", cv=10).mean(), 2))
    return(model)


# 2021 yılına ait weetlerin kurulmuş olan Logistic Regression modeli ile duygusunun tahmin edilmesi
def predict_new_tweet(dataframe_new, log_model, tf_idfVectorizer):
    tweet_tfidf = tf_idfVectorizer.transform(dataframe_new["tweet"])
    predictions = log_model.predict(tweet_tfidf)
    dataframe_new["label"] = predictions

    print("Tweet:", dataframe_new.loc[0, "tweet"])
    print("Prediction result :", dataframe_new.loc[0, "label"])
    print("-"*100)
    print("Tweet:", dataframe_new.loc[len(dataframe_new)-35, "tweet"])
    print("Prediction result :", dataframe_new.loc[len(dataframe_new)-1, "label"])
    return dataframe_new



def main_function():
    data = pd.read_csv("tweets_labeled.csv")
    print(data.head())
    print("-"*50)
    print(data.info())
    print("-"*50)
    print(data.columns)
    print("-"*50)
    print(data.shape)
    tf_idfVectorizer = TfidfVectorizer()
    X, y = data_preparation(data, tf_idfVectorizer)
    log_model = logistic_regression(X, y)
    dataframe_new = pd.read_csv("tweets_21.csv")
    predict_new_tweet(dataframe_new, log_model, tf_idfVectorizer)




print("The process has started.")
main_function()