# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Step 1: Dataset Import and Exploration
data_path = r"D:\computer class\prop\AIML_LAB\candy-data.csv"  # Path to the candy dataset
df = pd.read_csv(data_path)

# Create a binary target column based on a rule (e.g., candy popularity > median as '1', else '0')
df['target'] = (df['winpercent'] > df['winpercent'].median()).astype(int)

# Display basic details about the dataset
print("Basic details about the dataset:")
print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
print("Data types:\n", df.dtypes)
print("\nClass distribution:\n", df['target'].value_counts())  # Binary target distribution

# Step 2: Splitting dataset into features and target
X = df.drop(columns=['target', 'competitorname'])  # Drop 'competitorname' (non-numerical) and target
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 3: Define a function for model evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("AUC-ROC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Train a Logistic Regression model on the original dataset
print("\n--- Logistic Regression on Original Imbalanced Dataset ---")
original_model = LogisticRegression()
evaluate_model(original_model, X_train, y_train, X_test, y_test)

# Step 4: Handling Imbalanced Data
# Random Oversampling
minority_class = X_train[y_train == 1]
majority_class = X_train[y_train == 0]
minority_labels = y_train[y_train == 1]
majority_labels = y_train[y_train == 0]

X_train_over = pd.concat([majority_class, resample(minority_class, 
                                                   replace=True, 
                                                   n_samples=len(majority_class), 
                                                   random_state=42)])
y_train_over = pd.concat([majority_labels, resample(minority_labels, 
                                                    replace=True, 
                                                    n_samples=len(majority_class), 
                                                    random_state=42)])

print("\n--- Logistic Regression with Random Oversampling ---")
evaluate_model(original_model, X_train_over, y_train_over, X_test, y_test)

# Random Undersampling
X_train_under = pd.concat([resample(majority_class, 
                                    replace=False, 
                                    n_samples=len(minority_class), 
                                    random_state=42), minority_class])
y_train_under = pd.concat([resample(majority_labels, 
                                    replace=False, 
                                    n_samples=len(minority_class), 
                                    random_state=42), minority_labels])

print("\n--- Logistic Regression with Random Undersampling ---")
evaluate_model(original_model, X_train_under, y_train_under, X_test, y_test)

# SMOTE (Synthetic Minority Oversampling Technique)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\n--- Logistic Regression with SMOTE ---")
evaluate_model(original_model, X_train_smote, y_train_smote, X_test, y_test)

# Class Weighting
print("\n--- Logistic Regression with Class Weighting ---")
weighted_model = LogisticRegression(class_weight='balanced')
evaluate_model(weighted_model, X_train, y_train, X_test, y_test)

# Step 5: Summary of Results
print("\n--- Summary of Results ---")
print("Evaluate metrics such as Precision, Recall, F1-Score, and AUC-ROC from the outputs above.")
