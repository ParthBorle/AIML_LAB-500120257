# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Step 1: Dataset Import
data_path = r"D:\computer class\prop\AIML_LAB\candy-data.csv"  # Path to the candy dataset
df = pd.read_csv(data_path)

# Display basic details about the dataset
print("Dataset structure:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Step 2: Basic Visualizations (Matplotlib)

# Histogram: Distribution of winpercent
plt.figure(figsize=(8, 5))
plt.hist(df['winpercent'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Winpercent', fontsize=14)
plt.xlabel('Winpercent', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Bar Plot: Average winpercent by chocolate type
plt.figure(figsize=(8, 5))
avg_winpercent = df.groupby('chocolate')['winpercent'].mean()
avg_winpercent.plot(kind='bar', color=['skyblue', 'orange'], edgecolor='black')
plt.title('Average Winpercent by Chocolate Type', fontsize=14)
plt.xlabel('Chocolate', fontsize=12)
plt.ylabel('Average Winpercent', fontsize=12)
plt.xticks(rotation=0)
plt.show()

# Pie Chart: Proportion of candies with chocolate
plt.figure(figsize=(6, 6))
chocolate_counts = df['chocolate'].value_counts()
chocolate_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'orange'])
plt.title('Proportion of Candies with Chocolate', fontsize=14)
plt.ylabel('')  # Remove default ylabel
plt.show()

# Step 3: Advanced Visualizations (Seaborn)

# Box Plot: Distribution of winpercent by chocolate type
plt.figure(figsize=(8, 5))
sns.boxplot(x='chocolate', y='winpercent', data=df, palette='pastel')
plt.title('Winpercent Distribution by Chocolate Type', fontsize=14)
plt.xlabel('Chocolate', fontsize=12)
plt.ylabel('Winpercent', fontsize=12)
plt.show()

# Violin Plot: Distribution of winpercent by fruity type
plt.figure(figsize=(8, 5))
sns.violinplot(x='fruity', y='winpercent', data=df, palette='muted', split=True)
plt.title('Winpercent Distribution by Fruity Type', fontsize=14)
plt.xlabel('Fruity', fontsize=12)
plt.ylabel('Winpercent', fontsize=12)
plt.show()

# Pair Plot: Relationships among numerical features
sns.pairplot(df[['sugarpercent', 'pricepercent', 'winpercent']])
plt.suptitle('Pairwise Relationships', y=1.02, fontsize=16)
plt.show()

# Heatmap: Correlation matrix of numerical features
plt.figure(figsize=(10, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix', fontsize=16)
plt.show()

