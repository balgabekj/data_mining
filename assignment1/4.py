import pandas as pd

df = pd.read_csv('Health_Sleep_Statistics.csv')

df['Bedtime'] = pd.to_datetime(df['Bedtime'], format='%H:%M')
df['Wake-up Time'] = pd.to_datetime(df['Wake-up Time'], format='%H:%M')

def calculate_sleep_quality(row):

    bedtime_hour = int(row['Bedtime'].split(':')[0])
    late_bedtime_score = 1 if bedtime_hour >= 23 else 0

    sleep_disorders_score = 2 if row['Sleep Disorders'] == 'yes' else 0

    medication_score = 1 if row['Medication Usage'] == 'yes' else 0

    if row['Physical Activity Level'] == 'low':
        activity_score = -1
    elif row['Physical Activity Level'] == 'medium':
        activity_score = 0
    else:
        activity_score = 1

    dietary_score = 1 if row['Dietary Habits'] == 'unhealthy' else 0

    sleep_quality_score = 10 - late_bedtime_score - sleep_disorders_score - 0.5 * medication_score + 0.5 * activity_score - 0.5 * dietary_score
    return sleep_quality_score


df['Calculated Sleep Quality'] = df.apply(calculate_sleep_quality, axis=1)

print(df)
