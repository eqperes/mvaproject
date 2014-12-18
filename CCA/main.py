import CCA
import os
import numpy as np

import scipy.io as sio

import pickle

#X is the feature vector for the images
#T is the tag feature vector
#Y is the semantic class vector

#X=np.mat('[1 2 3 ; 3 4 5 ; 3 5 6 ; 3 6 7]')
#T=np.mat('[1 2 ; 4 5 ; 3 5 ; 4 6]')

img_features=sio.loadmat('../../layers /layer22/Flickr_8k.trainImages.mat')
feat=img_features['features']

X=np.zeros((feat.shape[0],feat[0,0].shape[2]))
for i in range(0,feat.shape[0]-1) :
    for j in range(0,feat[0,0].shape[2]-2) : 
        X[i,j]=feat[i,0][0,0,j]

T=pickle.load(open('../captions/image_features_lda_200_topics.dict','rb'))

ldafy = []
names = img_features["names"]
for name in names:
	ldafy.append(T[name[0][0]])
ldafy = np.array(ldafy)

[Wx,D]=CCA.CCA2(X,ldafy)
XX=np.concatenate((X,ldafy),axis=1)

#concatenated projection :
P=XX*Wx*D

#projected visual features : 
#P_visual=[P[:,1],P[:,2],...P[:,number_of_parameters_for_visual_feature]

#projected tag features :
#P_tag=[P[:,1],P[:,2],...P[:,number_of_parameters_for_tag_feature]

#then we can use CCA.NN(X,tag) to retrieve an image corresponding to a particular tag
