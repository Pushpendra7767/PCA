# import packages
import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from mpl_toolkits.mplot3d import Axes3D
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import	AgglomerativeClustering 

# print dataset & read
win = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\PCA\\wine.csv")
win.describe()
win.head()
# Considering only numerical data 
win.data = win.iloc[:,1:]
win.data.head(4)
# Normalizing the numerical data 
win_normal = scale(win.data)
pca = PCA(n_components = 6)
pca_values = pca.fit_transform(win_normal)
# The amount of variance that each PCA explains 
var = pca.explained_variance_ratio_
var
pca.components_[0]
# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1
# Variance plot for PCA components obtained 
plt.plot(var1,color="red")
# scatter plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:,2]
plt.scatter(x,y,color=["red"])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(np.array(x),np.array(y),np.array(z),c=["green"])

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)
# Normalized data frame (considering the numerical part of data)
win_norm = norm_func(win.iloc[:,1:])
win_norm.head(10)  # Top 10 rows

# K means clustering
# plot elbow curve 
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(win_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(win_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,win_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
# Selecting clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4) 
model.fit(win_norm)
model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
win['clust']=md # creating a  new column and assigning it to new column 
win_norm.head()

# hierarchical clustering
type(win_norm)
z = linkage(win_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# apply AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=4,	linkage='complete',affinity = "euclidean").fit(win_norm) 
cluster_labels=pd.Series(h_complete.labels_)
win['clust']=cluster_labels # creating a  new column and assigning it to new column 
win.head()























