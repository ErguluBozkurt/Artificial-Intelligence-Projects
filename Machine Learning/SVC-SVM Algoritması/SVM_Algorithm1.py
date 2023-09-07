"""
Bu projede position_salaries dataseti kullanılarak maaş tahmini yapılmıştır. Tahmin için SVM algoritması tercih edilmiştir.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

data = pd.read_csv('position_salaries.csv')
print(data.shape)

X = np.array(data["Level"]).reshape(-1,1)
y = np.array(data["Salary"]).reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

regressor = SVR(kernel='rbf', C=1)
regressor.fit(X, y)                    
                                   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plt.scatter(X, y, color="red")
plt.plot(X, regressor.predict(X), color="blue")
plt.title("SVR")
plt.xlabel("Pozisyon")
plt.ylabel("Maaş")
plt.show()

scorem = regressor.score(X_test, y_test)
print(f"Score: {round(scorem, 2)}")

sc_X_val = sc_X.transform(np.array([[6.5]]))
scaled_y_pred = regressor.predict(sc_X_val)
y_pred = sc_y.inverse_transform(scaled_y_pred.reshape(-1, 1))
print("6.5 Seviyesi Maaş:", float(y_pred))


