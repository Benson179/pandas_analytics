# pandas_analytics
# Task 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (replace with your own CSV file if needed)
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Display first few rows
print("First 5 rows:")
print(df.head())

# Check data types and missing values
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Clean dataset (drop rows with missing values)
df_clean = df.dropna()

# Task 2: Basic Data Analysis
print("\nBasic statistics:")
print(df_clean.describe())

# Group by species and compute mean of numerical columns
grouped = df_clean.groupby('species').mean(numeric_only=True)
print("\nMean values by species:")
print(grouped)

# Task 3: Data Visualization

# 1. Line chart (example: mean sepal length per species)
grouped['sepal_length'].plot(kind='line', marker='o')
plt.title('Mean Sepal Length per Species')
plt.xlabel('Species')
plt.ylabel('Mean Sepal Length')
plt.xticks(range(len(grouped.index)), grouped.index)
plt.show()

# 2. Bar chart (average petal length per species)
grouped['petal_length'].plot(kind='bar', color=['#4CAF50', '#2196F3', '#FFC107'])
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length')
plt.show()

# 3. Histogram (distribution of sepal width)
df_clean['sepal_width'].plot(kind='hist', bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot (sepal length vs. petal length)
plt.scatter(df_clean['sepal_length'], df_clean['petal_length'], c=df_clean['species'].astype('category').cat.codes, cmap='viridis', label=df_clean['species'])
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend(df_clean['species'].unique())
plt.show()

# Findings/Observations
print("\nObservations:")
print("- The dataset is clean with no missing values.")
print("- Setosa species has the smallest petal and sepal measurements on average.")
print("- There is a positive correlation between sepal length and petal length.")
