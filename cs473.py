import collections
import math
from nltk.stem import PorterStemmer
from collections import Counter

from scipy.spatial import distance

import numpy as np
import sys

PUNC = '''-!()[]{};:"\,—<>./?@#$%^&*_~|'''

PS = PorterStemmer()

def create_row(doc_words, all_word):
    row = []
    for key in all_word:
        if key in doc_words:
            row.append(doc_words[key])
        else:
            row.append(0)
    
    return row

		
# function to compute euclidean distance
def distance(p1, p2):
	return np.sum((p1 - p2)**2)

def get_cluster(erd, centroids, cen_d):
    erd_cen = centroids[0]
    min_d = math.inf
    for cen in centroids:
        dist = distance(cen, erd)
        if dist < min_d:
            erd_cen = cen_d[str(cen)]
            min_d = dist
    return erd_cen
    

# initialization algorithm
def initialize(data, k):
	
	## initialize the centroids list and add
	## a randomly selected data point to the list
	centroids = []
	centroids.append(data[np.random.randint(
			data.shape[0]), :])

	## compute remaining k - 1 centroids
	for c_id in range(k - 1):
		
		## initialize a list to store distances of data
		## points from nearest centroid
		dist = []
		for i in range(data.shape[0]):
			point = data[i, :]
			d = sys.maxsize
			
			## compute distance of 'point' from each of the previously
			## selected centroid and store the minimum distance
			for j in range(len(centroids)):
				temp_dist = distance(point, centroids[j])
				d = min(d, temp_dist)
			dist.append(d)
			
		## select data point with maximum distance as our next centroid
		dist = np.array(dist)
		next_centroid = data[np.argmax(dist), :]
		centroids.append(next_centroid)
		dist = []
		#print("centroid", centroids)
	return centroids

def call(mainDict, clusters):

    images = {} # dictionary = {"001.png" : {"flight" : 2.3, "end": 1.4}}
    
    # contains all words in all images in the folder, with text processing
    folderVocab = []

    # Dict with key: image name, value: Dict {key: word, value: score}
    imageWordscores = {}  

    for image in mainDict:
        imageString = ""  #long string containing all words in an image
        
        for object in mainDict[image]:
            #s now stores a long string containing words in one object of the image
            s = " ".join(mainDict[image][object]).lower()  

            #appends words in one object to imageString, which has words from all objects
            imageString += " " + s  

        for character in imageString:   # replace punctuation with a space
            if character in PUNC:
                imageString = imageString.replace(character, " ")
            
        doc_words = imageString.split(" ")  

        #for each word, stem it and add it to folderVocab if it is not already in it
        for key in doc_words:
            key = PS.stem(key)
            if key not in folderVocab:
                folderVocab.append(key)

        #wordFreq is a dictionary with key: word, value: term frequency     
        wordFreq = Counter(doc_words)
        wordFreq.pop('', None)

        print(wordFreq)
        
        #compute score for each word in the image
        for freq in wordFreq:
            wordFreq[freq] = 1 + math.log(wordFreq[freq])
        
        #Dict with key: image, value: wordFreq
        imageWordscores[image] = wordFreq

    all_words_mat = []
    
    for image in imageWordscores:
        row = np.array(create_row(imageWordscores[image], folderVocab))
        all_words_mat.append(row)
        images[str(row)] = image

    all_words_mat = np.array(all_words_mat)
    
    data = all_words_mat
    centroids = initialize(data, clusters)

    cen_d = {}
    for index, cen in enumerate(centroids):
        cen_d[str(cen)] = index

    file = open("base_line_clusters.txt", 'w')
    all_c = collections.defaultdict(list)

    
    for erd in data:   
        all_c[str(get_cluster(erd, centroids, cen_d))].append(images[str(erd)])

    out = ''
    for c in all_c:
        out += str(all_c[c]).strip('[').strip(']').replace("'", '').replace(".png", '') + '\n'
    
    file.write(out[:-1])

        

if __name__ == "__main__":
    file = open("parameters.txt", 'r')
    path_image = file.readline().strip("\n")
    clusters = file.readline().strip("\n")
    
    #call module 2, input is path_image, store its output in mainDict
    #call the call function with the dictionary returned by module 2


    # mainDict = {"001.png": {"entity": ["flighrt", "plane", "something"], "attribute": [
    #     "one"]}, 
    #     "002.png": {"entity": ["height", "plane", "something"], "attribute": ["two"]},
    #     "003.png": {"entity": ["random", "nonsense"], "attribute": ["three"]}
    #     }



    call(mainDict, int(clusters))
