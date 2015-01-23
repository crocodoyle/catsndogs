import theano
import theano.tensor as T

import os
import cv2

from sklearn.cluster import KMeans

import cPickle as pickle

#blog post __main__
if __name__ == "__main__":
    training_folder = 'C:/Data/train'
    
    cats = []
    dogs = []
    
    cat_kp = []
    dog_kp = []
    
    sift = cv2.SIFT()
    
    for root, dirs, files in os.walk(training_folder):
        for f in files:
            if 'cat' in f:
                cats.append(root + '/' + f)
            else:
                dogs.append(root + '/' + f)
                
    for c in cats[:1000]:
        print c
        cat = cv2.cvtColor(cv2.imread(c),cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(cat, None)
        
        for p in des:
            cat_kp.append(p)
    
        
    for d in dogs[:1000]:
        print d
        dog = cv2.cvtColor(cv2.imread(c),cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(dog, None)
        
        for p in des:
            cat_kp.append(p)
    
    cat_k_meaner = KMeans(n_jobs=-1, n_clusters=100)
    #dog_k_meaner = KMeans(n_jobs=-1, n_clusters=50)
    
    print 'k-means for cats...'
    cat_k_meaner.fit(cat_kp)
    print 'k-means for dogs...'
    #dog_k_meaner.fit(dog_kp)
    
    with open('C:/Data/centroids.pickle', 'wb') as handle:
        pickle.dump(cat_k_meaner.cluster_centers_, handle)
    #with open('dogs.pickle', 'wb') as handle:
        #pickle.dump(dog_k_meaner.cluster_centers_, handle)