import pandas as pd
import scipy.stats as stats
df = pd.read_csv('AppleStockDividend.csv')

df_no_dup = df.drop_duplicates()

print(df_no_dup.head())

df['Z_score'] = stats.zscore(df['Dividends'])

df_clean = df[(df['Z_score'] < 3) & (df['Z_score'] > -3)]
print(df.head())

df2 = pd.read_csv('Health_Sleep_Statistics.csv')

df2['Gender'] = df2['Gender'].replace({'m': 'male', 'f': 'female'})

print(df2.head())
