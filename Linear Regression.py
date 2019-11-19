import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats.mstats import gmean

pd.set_option('display.max_column', 10)
pd.set_option('display.max_row', 500)
pd.set_option('display.width', 500)
plt.style.use('ggplot')

df = pd.read_csv(r'C:\Users\Sudipto\Dropbox\Python\DA\datasets\BN.csv')
df[['Date', 'Expiry']] = df[['Date', 'Expiry']].apply(pd.to_datetime)
df.set_index('Date', inplace=True)
df['Range'] = df['High'] - df['Low']
df['CoC'] = df['Close'] - df['Underlying']
df['AvCoC'] = df['CoC'].rolling(3).apply(gmean, raw=True)
print(df)

plt.scatter(df['CoC'].shift(1), df['Close'], color='red')
plt.title('Close vs CoC')
plt.xlabel('CoC')
plt.ylabel('Close')
plt.show()