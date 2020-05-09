import time
import numpy as np
import glob
import cv2
import os
import shutil
import pyclustering

from pyclustering.cluster import xmeans
from pyclustering.cluster.encoder import cluster_encoder, type_encoding
from pyclustering.cluster.kmeans import kmeans
from scipy.spatial import distance
from sklearn.decomposition import PCA

TARGET_IMAGES_DIR = '../data/frames/'
CLUSTERED_IMAGES_DIR = '../data/CentroidImage/'
IMAGE_TYPE = 'png'
INITIAL_XMEANS_CENTERS = 10
PCA_COMPONENTS = 3

start_time = time.time()

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]

if not os.path.exists(TARGET_IMAGES_DIR):
    exit("Path do not exists")

filelist = glob.glob(TARGET_IMAGES_DIR + "*." + IMAGE_TYPE)

if len(filelist) < 1:
    exit("No images in this folder")

X = np.array([cv2.resize(cv2.imread(p), (64, 64), cv2.INTER_CUBIC) for p in filelist])
X = X.reshape(X.shape[0], -1)

pca = PCA(n_components = PCA_COMPONENTS)
pca.fit(X)
X_pca= pca.transform(X)

initializer = xmeans.kmeans_plusplus_initializer(data=X_pca, amount_centers=INITIAL_XMEANS_CENTERS)
initial_centers = initializer.initialize()
xm = xmeans.xmeans(data=X_pca, initial_centers=initial_centers)
xm.process()
clusters = xm.get_clusters()
centers = xm.get_centers()

centerIdx = list()

for idx, p in enumerate(centers):
    centerIdx.append(np.where(X_pca == closest_node(p, X_pca))[0][0])

if PCA_COMPONENTS < 4:
    ax = pyclustering.utils.draw_clusters(data=X_pca, clusters=clusters)

if os.path.exists(CLUSTERED_IMAGES_DIR):
    shutil.rmtree(CLUSTERED_IMAGES_DIR)
if not os.path.exists(CLUSTERED_IMAGES_DIR):
    os.makedirs(CLUSTERED_IMAGES_DIR)

label = 0
for c in clusters:
    for item in c:
        if (item == centerIdx[label]):
            shutil.copyfile(filelist[item], CLUSTERED_IMAGES_DIR + str(label) + "." + IMAGE_TYPE)
            label = label+1
            #print('Save', CLUSTERED_IMAGES_DIR + str(label) + "." + IMAGE_TYPE)
            break

print("--- execution time: %.2f seconds ---" % (time.time() - start_time))
