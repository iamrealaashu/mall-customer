# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import necessary libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load the Dataset
# Update the path if needed
import os

csv_path = "Task2/Mall_Customers.csv"
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
df = pd.read_csv(csv_path)

# Step 3: Explore the Data
print(df.head())
print(df.describe())

# Step 4: Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 5: Standardize the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Use Elbow Method to Find Optimal k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# Step 7: Train Final KMeans Model
k = 5  # use elbow method output to choose k
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 8: Visualize the Clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1', s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()