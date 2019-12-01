import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from matplotlib import pyplot as plt
import pickle
plt.style.use('ggplot')
pd.set_option('display.max_column', 5000)
pd.set_option('display.max_row', 5000)
pd.set_option('display.width', 5000)

df = pd.read_csv(r'C:\Users\Sudipto\Dropbox\Python\DA\datasets\BN.csv')
# df[['Date', 'Expiry']] = df[['Date', 'Expiry']].apply(pd.to_datetime)
df.sort_values('Date', inplace=True)
# df.set_index('Date', inplace=True)

df = df[['Expiry', 'Close', 'Prev Close HDFCBANK', 'Prev Close ICICIBANK', 'Prev Close AXISBANK',
         'Prev Close KOTAKBANK', 'Prev Close SBIN']]

print(df.head())

predict = 'Close'
x = np.array(df.drop([predict], 1))
y = np.array(df[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best = 0
for a in range(5000):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
        with open('topfive.pickle', 'wb') as f:
            pickle.dump(linear, f)

pickle_in = open('topfive.pickle', 'rb')

linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'Prev Close HDFCBANK'
plt.scatter(df[p], df['Close'])
plt.xlabel('HDFC')
plt.ylabel('BN Close')
plt.show()
