import pandas as pd
import yfinance as yf
import math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

df = yf.download("GOOGL", start="2004-08-19", end="2016-04-09")

df = df[['Close','High','Low','Open', 'Volume']]

df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100

#         price      x           x           x
df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

forecast_col = 'Close'
# df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
 
X = np.array(df.drop(['label'], axis=1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label']) 

X_train , X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
# with open('linearregresion.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# pickle_in = open('linearregresion.pickle', 'rb')
# clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
# print(accuracy)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

