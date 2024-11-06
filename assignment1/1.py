import pandas as pd

df = pd.read_csv('Iris.csv')

print('First 5 rows:\n', df.head())

print('\nLast 5 rows:\n', df.tail())

print('\nInfo:\n', df.info())

print("\nMissing values in each column:")
print(df.isnull())

print("\nData types of each column:")
print(df.dtypes)

