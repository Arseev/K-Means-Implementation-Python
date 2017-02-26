# Adam Seevers
# 01/2017
# Machine Learning
# Implementation of the K-means clustering algorithm.

import os
import numpy as np
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import csv
import sys
from tkinter import *

root = Tk()
#Have user select the csv file for analysis
root.fileName = filedialog.askopenfilename( filetypes = (("CSV Files", "*.csv"),))
print("File " + root.fileName + " has been loaded")
#Prompt user for the number of clusters they want to generate and the file they want to load
numberOfClusters = int(input("How many clusters are you testing?"))
#columns = int(input("How many attributes are in the dataset?"))
#numberOfAttributes = int(input("How many attributes does your data have?"))

data = np.genfromtxt(root.fileName, delimiter=',', skip_header=1, dtype=None)
#print(data)
instanceCount = (len(data))
def kmeans(data, km):
    #Iniitalize the centroid array
    centroids = []

    #Create the initial centroids based on random data points from instances
    centroids = generate_centroidsRND(data, centroids, km)  

    old_centroids = [[] for i in range(km)] 

    iterations = 0
    while not (has_converged(centroids, old_centroids, iterations)):
        iterations += 1

        clusters = [[] for i in range(km)]

        #assign data points to clusters based on euclidean distances
        clusters = euclidean_Dist(data, centroids, clusters)

        #recalculate centroids after each iteration for best cluster fit
        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            centroids[index] = np.mean(cluster).tolist()
            index += 1


    print("The total number of data instances is: " + str(instanceCount))
    print("The total number of iterations necessary is: " + str(iterations))
    print("The means for each cluster are: " + str(centroids))
    print("\n")
    clusterCount = 1;
    for cluster in clusters:
        print("cluster #" + str(clusterCount) + " is composed of " + str(len(cluster)) + " instances ")
        print(np.array(cluster).tolist())
        print("\n")
        clusterCount += 1
    return

#Randomly generate the initial centroids
def generate_centroidsRND(data, centroids, km):
    for cluster in range(km):
        #print(len(data))
        centroids.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
        
    return centroids

#Calculates the euclidean distance between an instances data points and all the cluster centroids and then finds the result in a 1 dimensional number      
def euclidean_Dist(data, centroids, clusters):
    for instance in data:  
        centroidIndex = min([(i[0], np.linalg.norm(instance-centroids[i[0]])) for i in enumerate(centroids)], key=lambda t:t[1])[0]
        #print(instance)
        try:
            clusters[centroidIndex].append(instance)
        except KeyError:
            clusters[centroidIndex] = [instance]

    #If any cluster is empty then assign one point from data set randomly so as to not have empty clusters       
    for cluster in clusters:
        if not cluster:
            cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
    return clusters

#Check if clusters have converged (i.e. stop iterating if clusters are defined to their max state)  
def has_converged(centroids, old_centroids, iterations):
    maxIterations = 1000
    if iterations > maxIterations:
        return True
    return old_centroids == centroids

kmeans(data, numberOfClusters);
