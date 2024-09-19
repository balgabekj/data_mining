import pandas as pd

df = pd.read_csv('AppleStockDividend.csv')

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df['RollingAvg_Dividends'] = df['Dividends'].rolling(window=3).mean()

df['ChangeInDividends'] = df['Dividends'].diff()

df['Year_Dividends_Interaction'] = df['Year'] * df['Dividends']


print(df.head())