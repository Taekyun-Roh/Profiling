#Import library 
from time import time
import numpy as np
import matplotlib.pyplot as plt
import os, sys


from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from matplotlib import style
style.use("ggplot")
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from pylab import *



input_file_path = './dataset/Driving dataset/Driving Data.csv'
file_open = open(input_file_path, 'r')

test_data_list = []
contents = ['']*9
count = int(0)


for tmpline in file_open.readlines():
	line = tmpline.rstrip('\r\n')
	tmp_list = line.split(',')
	count = count+1
	
	try:
		test_data_list.append(map(float, tmp_list))
	except:
		continue

XData = np.array(test_data_list)

#Demension Reduction 15->2
reduced_data = PCA(n_components=2).fit_transform(XData)



# K-Means Clustering
#-------------------------------------------------------------------------
cluster_num = 10
kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(reduced_data)
centroid = kmeans.cluster_centers_
labels = kmeans.labels_

#Figure initialize
c = Counter(labels)
fig = figure()
ax = fig.gca(projection='3d')

for n in range(len(reduced_data)):
	print("coordinate:", reduced_data[n], "label:", labels[n], "index:", n)
	#print "n : ", n
	#print "colors[labels[n]] : ", colors[labels[n]]
	ax.scatter(reduced_data[n][0],reduced_data[n][1])
	#print type(labels[n])
	
		
#Print Centroids	
print('-'*84)
print "centroid 0:"
print centroid[0]

print('-'*84)
print "centroid 1:"
print centroid[1]

print('-'*84)
print "centroid 2:"
print centroid[2]

print('-'*84)
print "centroid 3:"
print centroid[3]

print('-'*84)
print "centroid 4:"
print centroid[4]

print('-'*84)
print "centroid 5:"
print centroid[5]

print('-'*84)
print "centroid 6:"
print centroid[6]

print('-'*84)
print "centroid 7:"
print centroid[7]

print('-'*84)
print "centroid 8:"
print centroid[8]

print('-'*84)
print "centroid 9:"
print centroid[9]

print('-'*84)
print "total line"
print count


for cluster_number in range(cluster_num):
	print("Cluster {} contains {} samples".format(cluster_number, c[cluster_number]))

ax.scatter(centroid[:,0], centroid[:,1], marker = "X", s = 150, linewidths = 5, zorder = 100)
plt.show()