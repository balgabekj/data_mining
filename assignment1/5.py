import pandas as pd
import scipy.stats as stats
df = pd.read_csv('Health_Sleep_Statistics.csv')

df_no_dup = df.drop_duplicates()

print(df_no_dup.head())

# numeric_columns = ['Age', 'Sleep Quality', 'Daily Steps', 'Calories Burned']
#
# z_scores = stats.zscore(df[numeric_columns])
#
# z_score_df = pd.DataFrame(z_scores, columns=numeric_columns)
#
# print(z_score_df.head())

# df_no_outliers = df[(abs(z_score_df) < 3).all(axis=1)]
# print(df_no_outliers.head())


df['Gender'] = df['Gender'].replace({'m': 'male', 'f': 'female'})
print(df.head())
