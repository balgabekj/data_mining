from sklearn.preprocessing import MinMaxScaler

import pandas as pd

df = pd.read_csv('Health_Sleep_Statistics.csv')

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df[['Sleep Quality']])

df_scaled = pd.DataFrame(scaled_data, columns=['Sleep Quality'])
print('Scaled data: \n', df_scaled)

df_encoded = pd.get_dummies(df, columns=[ 'Dietary Habits', 'Sleep Disorders', 'Medication Usage'], drop_first=True)

print('Encoded data: \n', df_encoded.head())


df_new = df.copy()

age_bins = [0, 20, 30, 40, 50]
age_labels = ['<20', '20-30', '30-40', '40-50']

df_new['Age Group'] = pd.cut(df_new['Age'], bins=age_bins, labels=age_labels, right=False)

sleep_quality_bins = [0, 5, 7, 10]
sleep_quality_labels = ['Low', 'Medium', 'High']

df_new['Sleep Quality Group'] = pd.cut(df_new['Sleep Quality'], bins=sleep_quality_bins, labels=sleep_quality_labels, right=False)

print(df_new[['Age Group',  'Sleep Quality Group']].head())

# print(df.head(5))