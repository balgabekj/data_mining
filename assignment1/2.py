import pandas as pd

df = pd.read_csv('Iris.csv')

# # Identify missing values in the dataset using isnull().sum().
#
print('IdentifY missing values:', df.isnull().sum())

df_cleaned = df.dropna()
print('removed rows with missing values:')
print(df_cleaned)


# #fill missing values with mean value
df['SepalLengthCm'].fillna(df['SepalLengthCm'].mean(), inplace=True)
df['SepalWidthCm'].fillna(df['SepalWidthCm'].mean(), inplace=True)
df['PetalLengthCm'].fillna(df['PetalLengthCm'].mean(), inplace=True)
df['PetalWidthCm'].fillna(df['PetalWidthCm'].mean(), inplace=True)

# #fill missing values in 'Species' col with value 'Unknown'
df['Species'].fillna('Unknown', inplace=True)


df_ffill = df.ffill()
#
# # Display the DataFrame after forward fill
print(df_ffill)

df_bfill = df.bfill()
#
# Display the DataFrame after backward fill
print(df_bfill)

print(df)

