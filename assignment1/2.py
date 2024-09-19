import pandas as pd

df = pd.read_csv('Iris.csv')

print('IdentifY missing values:', df.isnull().sum())

df_cleaned = df.dropna()
print('removed rows with missing values:')
print(df_cleaned)


df['SepalLengthCm'].fillna(df['SepalLengthCm'].mean(), inplace=True)
df['SepalWidthCm'].fillna(df['SepalWidthCm'].mean(), inplace=True)
df['PetalLengthCm'].fillna(df['PetalLengthCm'].mean(), inplace=True)
df['PetalWidthCm'].fillna(df['PetalWidthCm'].mean(), inplace=True)

df['Species'].fillna('Unknown', inplace=True)


df_ffill = df.ffill()

#
print(df_ffill)

df_bfill = df.bfill()

#
print(df_bfill)

print(df)

