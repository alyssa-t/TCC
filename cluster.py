import numpy as np
import glob
import cv2
import os
import shutil
import pyclustering
from pyclustering.cluster import xmeans
from pyclustering.cluster.encoder import cluster_encoder, type_encoding
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer


from PIL import Image 
from sklearn import datasets
from sklearn.decomposition import PCA

TARGET_IMAGES_DIR = '/home/alyssa/TCC/Estudos/frames/'

filelist = glob.glob(TARGET_IMAGES_DIR+"*.png")
X = np.array([cv2.resize(cv2.imread(p), (64, 64), cv2.INTER_CUBIC) for p in filelist])
X = X.reshape(X.shape[0], -1)
#testar com as imagens em preto e branco dps

# 主成分分析前のサイズ
print(X.shape)
pca = PCA(n_components = 2)
pca.fit(X)
X_pca= pca.transform(X)
print(X_pca.shape)

initializer = xmeans.kmeans_plusplus_initializer(data=X_pca, amount_centers=2)
initial_centers = initializer.initialize()
xm = xmeans.xmeans(data=X_pca, initial_centers=initial_centers)
xm.process()
clusters = xm.get_clusters()


type_repr = xm.get_cluster_encoding();
encoder = cluster_encoder(type_repr, clusters, X_pca);
encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING);
centers = xm.get_centers()
print(centers)


kmeans_visualizer.show_clusters(X_pca, clusters, centers)
#ax = pyclustering.utils.draw_clusters(data=X_pca, clusters=clusters)

print("PCA累積寄与率: {}".format(sum(pca.explained_variance_ratio_)))



""""
for i in range(cluster_size):
    label = np.where(labels==i)[0]
    # Image placing
    if not os.path.exists(output_path+"/img_"+str(i)):
        os.makedirs(output_path+"/img_"+str(i))
        os.chown(output_path+"/img_"+str(i), uid, gid)
    for j in label:
        img = Image.open(img_paths[j])
        fname = img_paths[j].split('/')[-1]
        img.save(output_path+"/img_"+str(i)+"/" + fname)
        os.chown(output_path+"/img_"+str(i)+"/" + fname, uid, gid)
print("Image placing done.")
"""

print(xm.get_cluster_encoding())

# 主成分分析による次元削減

#X_pca= pca.transform(X)

# 主成分分析後のサイズ
#print(X_pca.shape)
label = 0
for c in clusters:
    if not os.path.exists('/home/alyssa/TCC/Estudos/frames/'+str(label)):
        os.makedirs('/home/alyssa/TCC/Estudos/frames/'+str(label))
    
    for item in c:
        shutil.copyfile(filelist[item], '/home/alyssa/TCC/Estudos/frames/'+str(label)+"/"+str(item)+".png")
    label = label+1