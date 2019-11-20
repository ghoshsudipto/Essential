import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

pd.set_option('display.max_column', 100)
pd.set_option('display.max_row', 500)
pd.set_option('display.width', 500)
plt.style.use('ggplot')

df = pd.read_csv(r'H:\Py\dataset\BN.csv')
df[['Date', 'Expiry']] = df[['Date', 'Expiry']].apply(pd.to_datetime)
df.set_index('Date', inplace=True)
df['RangeQ'] = (df['High'] - df['Low'])/3
 # df['High'] - df['Close'] < RangeQ

df['haha'] = np.select([df['High'].shift(1) - df['Close'].shift(1) < df['RangeQ']], [df['Close'].shift(1)-df['Close']])
# df['CoC'] = df['Close'] - df['Underlying']
# df['AvCoC'] = df['CoC'].rolling(3).mean()

condition = [(df['Expiry'] == df['Expiry'].shift(-1)) & (df['Expiry'] == df['Expiry'].shift(1)) & (df['Expiry'] == df['Expiry'].shift(2))]
choice = [df['Close'].rolling(3).mean()]
df['MA3'] = np.select(condition, choice)
df['MA3'].replace(0, np.nan, inplace=True)

print(df)

# plt.scatter(df['MA'], df['Close'], color='g')
# plt.title('Close vs MA')
# plt.xlabel('MA')
# plt.ylabel('Close')
# plt.show()

# plt.scatter(df['High'].shift(1), df['Close'], color='g')
# plt.title('Close vs MA')
# plt.xlabel('MA')
# plt.ylabel('Close')
# plt.show()
