# CSCI 3302: Homework 3 -- Clustering and Classification
# Implementations of K-Means clustering and K-Nearest Neighbor classification
import pickle
import random
import copy
import pdb
import matplotlib.pyplot as plt
from clustering_data import *
import math
import operator
import numpy as np

# TODO: INSERT YOUR NAME HERE
LAST_NAME = "Satchell"


def visualize_data(data, cluster_centers_file):
  fig = plt.figure(1, figsize=(4,3))
  f = open(cluster_centers_file, 'rb')
  centers = pickle.load(f)
  f.close()

  km = KMeansClassifier()
  km._cluster_centers = centers

  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

  labels = []
  center_colors = []
  for pt in data:
    labels.append(colors[km.classify(pt) % len(colors)])

  for i in range(len(centers)):
    center_colors.append(colors[i])

  plt.scatter([d[0] for d in data], [d[1] for d in data], c=labels, marker='x')
  plt.scatter([c[0] for c in centers], [c[1] for c in centers], c=center_colors, marker='D')
  plt.title("K-Means Visualization")
  plt.show()


class KMeansClassifier(object):

  def __init__(self):
    self._cluster_centers = [] # List of cluster centers, each of which is a point. ex: [ [10,10], [2,1], [0,-3] ]
    self._data = [] # List of datapoints (list of immutable lists, ex:  [ (0,0), (1.,5.), (2., 3.) ] )

  def add_datapoint(self, datapoint):
    self._data.append(datapoint)

  def fit(self, k):
    # Fit k clusters to the data, by starting with k randomly selected cluster centers.
    self._cluster_centers = [] # Reset cluster centers array

    # TODO: Initialize k cluster centers at random points
    # HINT: To choose reasonable initial cluster centers, you can set them to be in the same spot as random (different) points from the dataset

    i = 0
    for i in range (k):
      randNode = self._data[random.randint(0,len(self._data))]
      self._cluster_centers.append(randNode)
      i += 1
      # self._cluster_centers = [ self._data[random.randint(0,len(self._data))], #blue
      #                         self._data[random.randint(0,len(self._data))], #green
      #                         self._data[random.randint(0,len(self._data))], #red
      #                         self._data[random.randint(0,len(self._data))]  #cyan
                            # ] 
    # print("Cluster centers", self._cluster_centers, "\n")

    # TODO Follow convergence procedure to find final locations for each center
    classified_data = {}
    for i in range(0, k):
      classified_data[i] = []

    while True:
      # TODO: Iterate through each datapoint in self._data and figure out which cluster it belongs to
      # HINT: Use self.classify(p) for each datapoint p
      # print("data", self._data)
      for value in self._data:
        key = self.classify(value)
        classified_data[key].append(value)
      # print("classifed array", self._cluster_centers[0], "\n")
      # TODO: Figure out new positions for each cluster center (should be the average position of all its points)
      # print(self._data)
      # x_coord = 0
      # y_coord = 0
      centroid = []
      for i in range(0, k):
        centroid.append(np.mean(classified_data[i], axis = 0))
      
      # TODO: Check to see how much the cluster centers have moved (for the stopping condition)
      i = 0 
      while i < k:
        euclidean_distance = math.sqrt( (centroid[i][0]-self._cluster_centers[i][0])**2 + (centroid[i][1]-self._cluster_centers[i][1])**2 ) 
        # print("HERE", euclidean_distance)
      
      # TODO Add each of the 'k' final cluster_centers to the model (self._cluster_centers)
        # print("centroid before", centroid[i])
        self._cluster_centers[i] = centroid[i]
        # print("cluster_center", self._cluster_centers)
        i += 1
      
      # TODO: If the centers have moved less than some predefined threshold (you choose!) then exit the loop
      # smallest euclidean_distance < 0.0001:
      if euclidean_distance < 0.001:
        print("DONE!")
        break
      # else: 
      #   False

      # if True: # TODO: If the centers have moved less than some predefined threshold (you choose!) then exit the loop
      #   print("REACHED HERE", euclidean_distance)
      #   break      


  def classify(self,p):
    # Given a data point p, figure out which cluster it belongs to and return that cluster's ID (its index in self._cluster_centers)
    closest_cluster_index = 0

    # TODO Find nearest cluster center, then return its index in self._cluster_centers
    i = 0 
    min_dist = math.inf
    k = len(self._cluster_centers)
    # print("Cluster centers", len(self._cluster_centers))
    while i < k:
      euclidean_distance = math.sqrt( (p[0]-self._cluster_centers[i][0])**2 + (p[1]-self._cluster_centers[i][1])**2 )

      if euclidean_distance < min_dist:
        min_dist = euclidean_distance
        closest_cluster_index = i
      i += 1

    return closest_cluster_index

class KNNClassifier(object):

  def __init__(self):
    self._data = [] # list of (datapoint, label) tuples
  
  def clear_data(self):
    # Removes all data stored within the model
    self._data = []

  def add_labeled_datapoint(self, data_point, label):
    # Adds a labeled datapoint tuple onto the object's _data member
    self._data.append((data_point, label))

  def classify_datapoint(self, data_point, k):
    label_counts = {} # Dictionary mapping "label" => vote count
    best_label = None

    # Perform k_nearest_neighbor classification, setting best_label to the majority-vote label for k-nearest points
    #TODO: Find the k nearest points in self._data to data_point
    # print(self._data)
    label_dist = []

    for d in self._data:
      label = d[1]
      euclidean_distance = math.sqrt( (data_point[0]-d[0][0])**2 + (data_point[1]-d[0][1])**2 )
      label_dist.append([euclidean_distance, label])
    
    # Perform k_nearest_neighbor classification, setting best_label to the majority-vote label for k-nearest points
    #TODO: Find the k nearest points in self._data to data_point
    #TODO: Populate label_counts with the number of votes each label got from the k nearest points
    #TODO: Make sure to scale the weight of the vote each point gets by how far away it is from data_point
    #      Since you're just taking the max at the end of the algorithm, these do not need to be normalized in any way    
    
    k_size_count = sorted(label_dist)[:k]
    # print (sorted(k_size_count, key = lambda x: x[1]))
    
    for count in k_size_count:
      key = count[1]
      if key in label_counts.keys():
        prev = label_counts[key]
        label_counts[key] = prev + (1 / count[0])
      else:
        label_counts[key] = 1 / count[0] 
      # print(label_counts.get('a'))
    # for key,value in label_counts.items():
    #   print (key , " => " , value)
    
    best_label = max(label_counts.items(), key=operator.itemgetter(1))[0]
    # print(max(label_counts.items(), key=operator.itemgetter(1))[0])
    #max(stats.items(), key=operator.itemgetter(1))[0]

    return best_label



def print_and_save_cluster_centers(classifier, filename):
  for idx, center in enumerate(classifier._cluster_centers):
    print("  Cluster %d, center at: %s" % (idx, str(center)))


  f = open(filename,'wb')
  pickle.dump(classifier._cluster_centers, f)
  f.close()

def read_data_file(filename):
  f = open(filename)
  data_dict = pickle.load(f)
  f.close()

  return data_dict['data'], data_dict['labels']

def read_hw_data():
  global hw_data
  data_dict = pickle.loads(hw_data)
  return data_dict['data'], data_dict['labels']

def main():
  global LAST_NAME
  # read data file
  #data, labels = read_data_file('data.pkl')

  # load dataset
  data, labels = read_hw_data()

  # data is an 'N' x 'M' matrix, where N=number of examples and M=number of dimensions per example
  # data[0] retrieves the 0th example, a list with 'M' elements, one for each dimension (xy-points would have M=2)
  # labels is an 'N'-element list, where labels[0] is the label for the datapoint at data[0]


  ########## PART 1 ############
  # perform K-means clustering
  kMeans_classifier = KMeansClassifier()
  for datapoint in data:
    kMeans_classifier.add_datapoint(datapoint) # add data to the model

  kMeans_classifier.fit(4) # Fit 4 clusters to the data

  # plot results
  print('\n'*2)
  print("K-means Classifier Test")
  print('-'*40)
  print("Cluster center locations:")
  print_and_save_cluster_centers(kMeans_classifier, "kmeans_" + LAST_NAME + ".pkl")

  print('\n'*2)


  ########## PART 2 ############
  print("K-Nearest Neighbor Classifier Test")
  print('-'*40)

  # Create and test K-nearest neighbor classifier
  kNN_classifier = KNNClassifier()
  k = 4

  correct_classifications = 0
  # Perform leave-one-out cross validation (LOOCV) to evaluate KNN performance
  for holdout_idx in range(len(data)):
    # Reset classifier
    kNN_classifier.clear_data()

    for idx in range(len(data)):
      if idx == holdout_idx: continue # Skip held-out data point being classified

      # Add (data point, label) tuples to KNNClassifier
      kNN_classifier.add_labeled_datapoint(data[idx], labels[idx])

    guess = kNN_classifier.classify_datapoint(data[holdout_idx], k) # Perform kNN classification

    if guess == labels[holdout_idx]: 
      correct_classifications += 1.0
  
  print("kNN classifier for k=%d" % k)
  print("Accuracy: %g" % (correct_classifications / len(data)))
  print('\n'*2)

  visualize_data(data, 'kmeans_' + LAST_NAME + '.pkl')


if __name__ == '__main__':
  main()
