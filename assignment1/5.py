import pandas as pd
import scipy.stats as stats
df = pd.read_csv('Health_Sleep_Statistics.csv')

df_no_dup = df.drop_duplicates()

z_score = stats.zscore(df[['Age', 'Sleep Quality', 'Daily Steps', 'Calories Burned']])
df_no_outliers = df[(z_score < 3).all(axis=1)]
print(df_no_outliers.head())

df['Gender'] = df['Gender'].replace({'m': 'male', 'f': 'female'})

print(df.head())
