# Imtiaz- Import Libraries
from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
# Imtiaz- Upload Datasets
uploaded = files.upload()  # upload dataset1.csv + dataset2.csv
d1 = pd.read_csv("dataset1.csv")
d2 = pd.read_csv("dataset2.csv")
# Imtiaz- Clean columns
d1.columns = d1.columns.str.lower().str.strip()
d2.columns = d2.columns.str.lower().str.strip()
# Imtiaz- Dataset overview
print("Dataset1 shape:", d1.shape)
print("Dataset2 shape:", d2.shape)
print(d1.head(), "\n")
print(d2.head())
# Shazid- Descriptive stats
print(d1.describe())
print(d2.describe())
# Shazid- Distribution of risk-taking behaviour
sns.countplot(x='risk', data=d1)
plt.title("Bat behaviour: Risk-taking vs Risk-avoidance")
plt.show()
# Shazid- Risk vs Reward
sns.countplot(x='risk', hue='reward', data=d1)
plt.title("Risk vs Reward outcomes")
plt.show()