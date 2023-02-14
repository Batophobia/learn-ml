import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

data = pd.read_csv("Notebooks/17-K-Means-Clustering/College_Data", index_col=0)
data["Grad.Rate"]["Cazenovia College"] = 100
#print(data.head())
#sns.scatterplot(data=data, x="Grad.Rate", y="Room.Board", hue="Private")
#sns.scatterplot(data=data, x="F.Undergrad", y="Outstate", hue="Private")
#g = sns.FacetGrid(data, hue="Private")
#g = g.map(plt.hist, 'Outstate', bins=30, alpha=0.65)
#g = g.map(plt.hist, 'Grad.Rate', bins=30, alpha=0.65)

#print(data[data["Grad.Rate"] > 100])
kmeans = KMeans(n_clusters=2)
kmeans.fit(data.drop('Private', axis=1))
#print(kmeans.cluster_centers_)

data["Cluster"] = data["Private"].apply(converter)
#print(data.head())
print(confusion_matrix(data['Cluster'], kmeans.labels_))
print(classification_report(data['Cluster'], kmeans.labels_))
#plt.show()