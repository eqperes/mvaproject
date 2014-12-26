import os
import numpy as np

import scipy.io as sio
from CCAbib import CCA2view
import pickle


#X is the feature vector for the images
#T is the tag feature vector
#Y is the semantic class vector

#X=np.mat('[1 2 3 ; 3 4 5 ; 3 5 6 ; 3 6 7]')
#T=np.mat('[1 2 ; 4 5 ; 3 5 ; 4 6]')

img_features=sio.loadmat('../../../layers /layer21/Flickr_8k.trainImages.mat')
feat=img_features['features']

X=np.zeros((feat.shape[0],feat[0,0].shape[2]))
for i in range(0,feat.shape[0]-1) :
    for j in range(0,feat[0,0].shape[2]-2) : 
        X[i,j]=feat[i,0][0,0,j]

T=pickle.load(open('../captions/image_features_lda_200_topics.dict','rb'))

lda_features_train = []
names = img_features["names"]
for name in names:
	lda_features_train.append(T[name[0][0]])
lda_features_train = np.array(lda_features_train)

my_cca = CCA2view(200)
my_cca.fit(X, lda_features_train)

img_features_test=sio.loadmat('../../../layers /layer21/Flickr_8k.testImages.mat')
test_feat=img_features_test['features']

X_test=np.zeros((test_feat.shape[0],test_feat[0,0].shape[2]))
for i in range(0,test_feat.shape[0]-1) :
    for j in range(0,test_feat[0,0].shape[2]-2) : 
        X_test[i,j]=test_feat[i,0][0,0,j]

lda_features_test = []
names = img_features_test["names"]
for name in names:
	lda_features_test.append(T[name[0][0]])
lda_features_test = np.array(lda_features_test)

matching_scores = my_cca.matching_score(X_test, lda_features_test)

[samples, dim] = lda_features_test.shape

random_scores_matrix = np.zeros((samples, samples))

for i in range(0,samples):
	X_dummy = np.tile(X_test[i], (samples, 1))
	random_scores_matrix[i] = my_cca.matching_score(X_dummy, lda_features_test)

random_scores = np.sum(random_scores_matrix, axis=0)

random_scores = (float(1)/float(samples))*random_scores




