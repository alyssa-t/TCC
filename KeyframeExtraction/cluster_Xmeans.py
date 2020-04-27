import numpy as np
import glob
import cv2
import os
import shutil
import pyclustering

from pyclustering.cluster import xmeans
from pyclustering.cluster.encoder import cluster_encoder, type_encoding
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from scipy.spatial import distance
from sklearn.decomposition import PCA

TARGET_IMAGES_DIR = '../data/frames/'
CENTROID_IMAGES_DIR = '../data/CentroidImage/'
IMAGE_TYPE = 'png'
INITIAL_XMEANS_CENTERS = 3
PCA_COMPONENTS = 2

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


filelist = glob.glob(TARGET_IMAGES_DIR + "*." + IMAGE_TYPE)

X = np.array([cv2.resize(cv2.imread(p), (64, 64), cv2.INTER_CUBIC) for p in filelist])
X = X.reshape(X.shape[0], -1)

pca = PCA(n_components = PCA_COMPONENTS)
pca.fit(X)
X_pca= pca.transform(X)
pca = X_pca/ np.sqrt(np.sum(X_pca**2))

initializer = xmeans.kmeans_plusplus_initializer(data=X_pca, amount_centers=INITIAL_XMEANS_CENTERS)
initial_centers = initializer.initialize()
xm = xmeans.xmeans(data=X_pca, initial_centers=initial_centers)
xm.process()
clusters = xm.get_clusters()
centers = xm.get_centers()

centerIdx = list()
print (centerIdx)
for idx, p in enumerate(centers):
    centerIdx.append(np.where(X_pca == closest_node(p, X_pca))[0][0])
print (centerIdx)
print(len(clusters))
print (clusters)

if PCA_COMPONENTS < 4:
    ax = pyclustering.utils.draw_clusters(data=X_pca, clusters=clusters)
    #kmeans_visualizer.show_clusters(X_pca, clusters, centers)


if os.path.exists(CENTROID_IMAGES_DIR):
    shutil.rmtree(CENTROID_IMAGES_DIR)
if not os.path.exists(CENTROID_IMAGES_DIR):
    os.makedirs(CENTROID_IMAGES_DIR)

label = 0
for c in clusters:
    for item in c:
        if (item == centerIdx[label]):
            shutil.copyfile(filelist[item], CENTROID_IMAGES_DIR + str(label) + "." + IMAGE_TYPE)
            label = label+1
            print('Save', CENTROID_IMAGES_DIR + str(label) + "." + IMAGE_TYPE)
            break


