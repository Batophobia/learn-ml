import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()
#print(data.keys())
df = pd.DataFrame(data["data"], columns=data["feature_names"])

scaler = StandardScaler()
scaler.fit(df)
scaleData = scaler.transform(df)

pca = PCA(n_components=2)
pca.fit(scaleData)

x_pca = pca.transform(scaleData)
#print(scaleData.shape)
#print(x_pca.shape)

#plt.scatter(x_pca[:,0], x_pca[:,1], c=data['target'])

dfComp = pd.DataFrame(pca.components_, columns=data['feature_names'])
sns.heatmap(dfComp)

plt.show()
