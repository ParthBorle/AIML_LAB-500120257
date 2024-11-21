
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


file_path = "D:\computer class\prop\AIML_LAB\Customers (1).csv"  # Update with your file path
df = pd.read_csv(file_path)

print("Dataset Info:")
print(df.info())  
print("\nFirst 5 Rows:\n", df.head())  


numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=[object]).columns
 

df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.mean()))
df[categorical_cols] = df[categorical_cols].apply(lambda col: col.fillna("Unknown"))
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(df[numeric_cols])

if len(categorical_cols) > 0:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_encoded = encoder.fit_transform(df[categorical_cols])
else:
    categorical_encoded = np.array([])

if categorical_encoded.size > 0:
    final_features = np.hstack((numeric_scaled, categorical_encoded))
else:
    final_features = numeric_scaled

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(final_features)

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c='blue', alpha=0.6)
plt.title("PCA Scatter Plot")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(reduced_features)

df['Cluster'] = clusters

plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title("K-Means Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

print("\nCluster Distribution:")
print(df['Cluster'].value_counts())

sns.countplot(x='Cluster', data=df)
plt.title("Cluster Distribution")
plt.show()
