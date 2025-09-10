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

#Al-Amin Dhaly Part Starts

#Boxplot

sns.boxplot(x='risk', y='bat_landing_to_food', data=d1)
plt.title("Landing-to-food time by risk")
plt.show()

# Histogram: seconds after rat arrival
sns.histplot(d1['seconds_after_rat_arrival'].dropna(), bins=30, kde=True)
plt.title("Seconds after rat arrival")
plt.show()

#Inferential analysis
# t-test: landing_to_food between risk vs no risk
risk_grp = d1[d1['risk']==1]['bat_landing_to_food'].dropna()
no_risk_grp = d1[d1['risk']==0]['bat_landing_to_food'].dropna()
t, p = stats.ttest_ind(risk_grp, no_risk_grp)
print("t-test landing_to_food risk vs no risk:", t, p)

# Logistic regression: seconds_after_rat_arrival predicting risk
X = sm.add_constant(d1[['seconds_after_rat_arrival']].fillna(0))
y = d1['risk']
logit = sm.Logit(y, X).fit(disp=False)
print(logit.summary())

#Scatter: bat vs rat arrivals
sns.scatterplot(x='rat_arrival_number', y='bat_landing_number', data=d2)
plt.title("Bat landings vs Rat arrivals")
plt.show()

# Correlation
corr = d2['rat_arrival_number'].corr(d2['bat_landing_number'])
print("Correlation (bat vs rat arrivals):", corr)

#Correlation heatmap for Dataset1
plt.figure(figsize=(10,6))
sns.heatmap(d1.corr(numeric_only=True), annot=True, cmap="Greens", fmt=".2f")
plt.title("Correlation heatmap - Dataset1 (Bat landings)")
plt.show()

#Correlation heatmap for Dataset2
plt.figure(figsize=(8,6))
sns.heatmap(d2.corr(numeric_only=True), annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation heatmap - Dataset2 (Rat arrivals)")
plt.show()

#Al-Amin Dhaly Ends