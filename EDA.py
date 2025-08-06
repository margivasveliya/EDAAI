

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


file_name = "data.csv"  
data = pd.read_csv(file_name)  
print("Data loaded successfully!")


print("\n First 5 rows of data:")
print(data.head())  

print("\n Number of rows and columns:", data.shape)
print("\n Names of columns:", data.columns.tolist())

print("\n Data types of each column:")
print(data.dtypes)

print("\n Missing values in each column:")
print(data.isnull().sum())


print("\n Summary statistics (mean, std, min, max, etc.):")
print(data.describe())  

print("\n Median values for each numeric column:")
print(data.median(numeric_only=True))


numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    plt.figure()
    sns.histplot(data[col], kde=True, color="skyblue")
    plt.title(f" Histogram for {col}")
    plt.show()


for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=data[col], color="orange")
    plt.title(f" Boxplot for {col}")
    plt.show()


sns.pairplot(data[numeric_cols], diag_kind="kde", plot_kws={"alpha": 0.6})
plt.suptitle(" Pairplot of Numeric Features", y=1.02)
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title(" Correlation Heatmap")
plt.show()


if len(numeric_cols) >= 2:
    fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1],
                     title=f"Interactive Scatter: {numeric_cols[0]} vs {numeric_cols[1]}")
    fig.show()


print("\n My Observations:")
for col in numeric_cols:
    print(f"- {col}: mean = {data[col].mean():.2f}, std = {data[col].std():.2f}")
    
   
    if data[col].isnull().sum() > 0:
        print(f" Missing values found in {col}")
    
    
    if data[col].skew() > 1 or data[col].skew() < -1:
        print(f"   {col} looks highly skewed - maybe needs transformation later")

print("\n EDA finished! You can now explore your data better.")
