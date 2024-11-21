# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Dataset Import and Preprocessing
data_path = r"D:\computer class\prop\AIML_LAB\candy-data.csv"  # Path to the candy dataset
df = pd.read_csv(data_path)

# Display dataset structure and first few rows
print("Dataset structure:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Checking for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Selecting features and target
X = df[['sugarpercent', 'pricepercent']]  # Features for regression
y = df['winpercent']  # Target variable

# Step 2: Exploratory Data Analysis (EDA)
# Scatter plot to visualize relationships
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sugarpercent', y='winpercent', data=df, color='blue', label='Sugar Percent')
plt.title('Sugar Percent vs Winpercent')
plt.xlabel('Sugar Percent')
plt.ylabel('Winpercent')
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='pricepercent', y='winpercent', data=df, color='orange', label='Price Percent')
plt.title('Price Percent vs Winpercent')
plt.xlabel('Price Percent')
plt.ylabel('Winpercent')
plt.legend()
plt.show()

# Heatmap to visualize correlations (for numeric columns only)
plt.figure(figsize=(10, 6))
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix', fontsize=16)
plt.show()

# Step 3: Linear Regression Model Implementation
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Step 4: Evaluation Metrics
# Calculating metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Value: {r2:.2f}")

# Step 5: Visualizing the Regression Line
# Plotting actual vs. predicted values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='purple', alpha=0.6)
plt.title('Actual vs Predicted Winpercent')
plt.xlabel('Actual Winpercent')
plt.ylabel('Predicted Winpercent')
plt.axline([0, 0], [1, 1], color='red', linestyle='--', label='Ideal Prediction')
plt.legend()
plt.show()

# Regression line for sugarpercent (single-feature regression example)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_train['sugarpercent'], y=y_train, color='blue', alpha=0.6, label='Training Data')
plt.plot(X_train['sugarpercent'], model.predict(X_train), color='red', label='Regression Line')
plt.title('Regression Line: Sugar Percent vs Winpercent')
plt.xlabel('Sugar Percent')
plt.ylabel('Winpercent')
plt.legend()
plt.show()

