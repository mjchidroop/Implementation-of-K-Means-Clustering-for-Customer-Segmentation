# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialization: Select k cluster centroids (means), either randomly or using a smart initialization.

2. Assignment: Each customer’s data point is assigned to the nearest centroid, forming cluster memberships.

3. Update: Recalculate each centroid as the mean of all points assigned to that cluster.

4. Repeat: Assignment and update steps repeat until centroids stabilize (no significant change), or a maximum number of iterations is reached.

5. Output: Customers are segmented into k distinct clusters representing similar spending and income patterns.

## Program:
```py
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: CHIDROOP M J
RegisterNumber:  25018548
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data
data = pd.read_csv('Mall_Customers.csv')

# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow method - find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Apply KMeans with optimal clusters (k=5 for this dataset)
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(
        X.values[y_kmeans == i, 0],
        X.values[y_kmeans == i, 1],
        s=50,
        c=colors[i],
        label=f'Cluster {i}'
    )
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='yellow',
    label='Centroids'
)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments (K Means)')
plt.legend()
plt.show()

```

## Output:
<img width="461" height="646" alt="Screenshot 2025-10-05 161328" src="https://github.com/user-attachments/assets/c77eea59-6510-4aeb-87cc-538d5d6a1d61" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
