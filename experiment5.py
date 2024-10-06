import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

# Function to load the dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Function to export the dataset
def export_dataset(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Dataset successfully exported to {file_path}")

# Function to perform dataset details for EDA
def dataset_details(df):
    # Basic information
    print("Basic Information about the Dataset:")
    print(df.info())

    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Checking missing values
    print("\nMissing Values per Column:")
    print(df.isnull().sum())

    # Checking unique values per column
    print("\nUnique Values per Column:")
    print(df.nunique())

    # Pairplot of the dataset (for numerical columns)
    print("\nPairplot for Numerical Columns:")
    sns.pairplot(df)
    plt.show()

    # Heatmap for missing values
    print("\nHeatmap for Missing Values:")
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.show()

    # Correlation matrix
    print("\nCorrelation Matrix:")
    correlation_matrix = df.corr(numeric_only=True)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()

    # Plotting histograms for numerical columns
    print("\nHistograms for Numerical Columns:")
    df.hist(figsize=(10, 8), bins=20)
    plt.tight_layout()
    plt.show()

# Load the dataset with the specified file path
file_path = 'D:/computer class/prop/AIML_LAB/databasee.csv'
df = load_dataset(file_path)

# Perform EDA
dataset_details(df)
