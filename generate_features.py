# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 14:11:54 2015

@author: Andrew
"""

import theano
import theano.tensor as T

import os
import cv2

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from sklearn.naive_bayes import GaussianNB

import cPickle as pickle
from operator import itemgetter
import numpy as np

import csv


def load_training_features(image_dir):
    cat_features = pickle.load(open('C:/Data/cat_features.pickle', 'rb'))
    dog_features = pickle.load(open('C:/Data/dog_features.pickle', 'rb'))

    return cat_features, dog_features

def load_testing_features(image_dir):
    features = pickle.load(open(image_dir + 'test_features.pickle', 'rb'))
    return features

def load_centroids(filename):
    centroids = pickle.load(open('C:/Data/centroids.pickle', 'rb'))
    
    return centroids
    
def build_features(filename, centroids):
    print filename
    image = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2GRAY)
    kp, descriptors = sift.detectAndCompute(image, None)

    feature_vector = np.zeros(np.shape(centroids)[0])
    for des in descriptors:
        
        #distance of each sift feature to centroids
        ssd = np.zeros(np.shape(centroids)[0])
        
        for i, cent in enumerate(centroids):
            ssd[i] = np.sum((des - cent)**2)
        
        #count each sift feature as its closest centroid
        feature_vector[min(enumerate(ssd), key=itemgetter(1))[0]] += 1         
        
    #normalize so that feature vector in centroid space sums to 1
    normalized_features = feature_vector
     
    return normalized_features    

def generate_training_features(image_dir, centroids):
    images = []
    
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            images.append(root + f)

    cat_features = np.zeros((len(images), np.shape(centroids)[0]))
    dog_features = np.zeros((len(images), np.shape(centroids)[0]))

    for img in images:
        index = img[img.find('.')+1:img.rfind('.')]
        if 'cat' in img:
            cat_features[index,:] = build_features(img, centroids)
        if 'dog' in img:
            dog_features[index,:] = build_features(img, centroids)
    
    with open('C:/Data/cat_features.pickle', 'wb') as handle:
        pickle.dump(cat_features, handle)
    with open('C:/Data/dog_features.pickle', 'wb') as handle:
        pickle.dump(dog_features, handle)
    
    return [cat_features, dog_features]

def generate_testing_features(image_dir, centroids):
    images = []
    
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            images.append(root + f)

    features = np.zeros((len(images)+1, np.shape(centroids)[0]))
    
    for img in images:
        index = img[img.rfind('/')+1:img.rfind('.')]
        try:
            features[index,:] = build_features(img, centroids)
        except:
            pass
    with open('C:/Data/test_features.pickle', 'wb') as handle:
        pickle.dump(features, handle)
        
    return features


def test_model(cat_features, dog_features, test_features):
    Y1 = np.zeros((np.shape(cat_features)[0])) 
    Y2 = np.ones((np.shape(dog_features)[0]))
        
    
    Y = np.hstack((Y1, Y2)).transpose()
    x = np.vstack((cat_features, dog_features))
 
    print np.shape(Y)
    print np.shape(x)
 
    gnb = GaussianNB()
    gnb.fit(x, Y)
    
    y_predictions = gnb.predict(test_features)
    
    with open('C:/Data/test_submission.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id', 'label'])        
        for i, line in enumerate(y_predictions):
            print line
            writer.writerow([i, line])

if __name__ == "__main__":
    #centroids = load_centroids('C:/Data/train/centroids.pickle')
    
    #cats, dogs = generate_training_features('C:/Data/train/', centroids)
    cats, dogs = load_training_features('C:/Data/train/')    
    #testing_features = generate_testing_features('C:/Data/test/', centroids)
    testing_features = load_testing_features('C:/Data/')
    
    test_model(cats, dogs, testing_features)