import cv2
import numpy as np
import csv
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error, davies_bouldin_score, silhouette_score, pairwise_distances, calinski_harabasz_score


def dunn_index(X, labels):
	# Get unique clusters
	clusters = np.unique(labels)
	n_clusters = len(clusters)

	# Calculate pairwise distances
	dist_matrix = pairwise_distances(X)

	# Maximum intra-cluster distance
	max_intra_distance = 0
	for cluster in clusters:
		cluster_points = X[labels == cluster]
		if len(cluster_points) > 1:
			intra_distances = dist_matrix[labels == cluster][:, labels == cluster]
			max_intra_distance = max(max_intra_distance, intra_distances.max())

	# Minimum inter-cluster distance
	min_inter_distance = np.inf
	for i in range(n_clusters):
		for j in range(i + 1, n_clusters):
			inter_distances = dist_matrix[labels == clusters[i]][:, labels == clusters[j]]
			min_inter_distance = min(min_inter_distance, inter_distances.min())

	# Calculate Dunn Index
	if max_intra_distance == 0:
		return 0
	return min_inter_distance / max_intra_distance



def calculateClusteringPerformanceMeasures(X, label, u_labels, centroids, perfMeasuresFileName, rowForDataframe, dfResults, runtime):
	centroidsList = []
	for l in label:
		centroidsList.append(list(centroids[l]))

	# print("X:")
	# print(X)
	# print("label")
	# print(label)
	# print("centroids")
	# print(centroids)
	# print("centroidsList")
	# print(centroidsList)

	
	groundTruth = []
	for x in X:
		groundTruth.append([x[0],x[1]])
	
	# print("gt")
	# print(groundTruth)
	# print("gt length: " + str(len(groundTruth)))

	d = {"groundTruth": groundTruth, "cluster": label, "centroid": centroidsList}
	df = pd.DataFrame(d)
	# print(df["groundTruth"])


	mse = mean_squared_error(df["groundTruth"].to_list(), df["centroid"].to_list())
	#print(f'mse = {mse}')
	#rowForDataframe["MSE"] = [mse]
	rowForDataframe.append(mse)

	daviesBouldinScore = davies_bouldin_score(X, label)
	#print(f'DBI = {daviesBouldinScore}')
	# rowForDataframe["DBI"] = [daviesBouldinScore]
	rowForDataframe.append(daviesBouldinScore)

	silhouetteScore = silhouette_score(X,label)
	#print(f'Silhouette Score = {silhouetteScore}')
	# rowForDataframe["silhouetteScore"] = [silhouetteScore]
	rowForDataframe.append(silhouetteScore)
	
	dunnIndex = dunn_index(X,label)
	#print(f'Dunn Index =  {dunnIndex}')

	chIndex = calinski_harabasz_score(X, label)
	#print(f'Calinsky-Harabasz = {chIndex}')
	# rowForDataframe["chIndex"] = [chIndex]
	rowForDataframe.append(chIndex)
	
	rowForDataframe.append(runtime)
	dfResults.loc[len(dfResults.index)] = rowForDataframe
	# print(dfResults)
	# input()

	#d = [{"mse": mse, "DBI": daviesBouldinScore, "Silhouette Score": silhouetteScore, "Calinsky-Harabasz":chIndex}]
	# d = [{"DBI": daviesBouldinScore, "Silhouette Score": silhouetteScore, "Calinsky-Harabasz":chIndex}]

	# df = pd.DataFrame(d)
	# print(df)
	# df.to_csv(perfMeasuresFileName)
	
	




