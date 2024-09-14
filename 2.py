import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Iris.csv")

statistics = df.describe()

# correlation_matrix = df.corr()

print(statistics)