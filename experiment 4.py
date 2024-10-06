import pandas as pd
import os

# Path to the CSV file
file_path = 'D:\computer class\prop\AIML_LAB\databasee.csv'  # Modify this to the correct file path if needed

# Check if the file exists before proceeding
if os.path.exists(file_path):
    # Import data from a CSV file
    df = pd.read_csv(file_path)
    
    # Display dataset details
    print("Number of rows:", df.shape[0])  # Number of rows
    print("Number of columns:", df.shape[1])  # Number of columns

    # Display first five rows
    print("\nFirst five rows:\n", df.head())
    
    # Size of the dataset
    print("\nSize of the dataset:", df.size)
    
    # Checking for missing values in the dataset
    print("\nNumber of missing values:\n", df.isnull().sum())

    # Display summary statistics for numerical columns
    print("\nSum of numerical columns:\n", df.sum(numeric_only=True))
    print("\nAverage of numerical columns:\n", df.mean(numeric_only=True))
    print("\nMinimum values of numerical columns:\n", df.min(numeric_only=True))
    print("\nMaximum values of numerical columns:\n", df.max(numeric_only=True))

    # Export data to a new CSV file
    df.to_csv('output.csv', index=False)
    print("\nData exported successfully to 'output.csv'")
else:
    print(f"Error: File '{file_path}' not found.")
