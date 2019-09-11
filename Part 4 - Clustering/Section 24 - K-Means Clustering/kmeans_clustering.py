
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values

# using elbow method to find th optimal number of clusters

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.plot(range(1,11),wcss)
plt.title('elbow method')

kmeans=KMeans(n_clusters=5,init='k-means++')
y_kmeans=kmeans.fit_predict(X)

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,color='red',label='careless')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,color='green',label='garib')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,color='blue',label='baniye')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,color='orange',label='ayisahhh')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,color='black',label='middle class')
plt.title('clusters')
plt.xlabel('annual income')
plt.ylabel('spendiong score')
plt.legend()
plt.show()