import os
import numpy as np

import scipy.io as sio
import CCA
import pickle

#X is the feature vector for the images
#T is the tag feature vector
#Y is the semantic class vector

#X=np.mat('[1 2 3 ; 3 4 5 ; 3 5 6 ; 3 6 7]')
#T=np.mat('[1 2 ; 4 5 ; 3 5 ; 4 6]')

img_features=sio.loadmat('../../../layers /layer22/Flickr_8k.trainImages.mat')
feat=img_features['features']

X=np.zeros((feat.shape[0],feat[0,0].shape[2]))
for i in range(0,feat.shape[0]-1) :
    for j in range(0,feat[0,0].shape[2]-2) : 
        X[i,j]=feat[i,0][0,0,j]

T=pickle.load(open('../captions/image_features_lda_200_topics.dict','rb'))

ldaf = []
names = img_features["names"]
for name in names:
	ldaf.append(T[name[0][0]])
ldaf = np.array(ldaf)

[Wx,D]=CCA.CCA2(X,ldaf)
XX=np.concatenate((X,ldaf),axis=1)

# my_cca = CCA(n_components=15)
# my_cca.fit(X, ldaf)

# all_scores = np.zeros((names.shape[0], names.shape[0]))
# for k in range(0, names.shape[0]):
# 	j = 0
# 	for name in names:
# 		all_scores[k][j] = my_cca.score(X[k], T[name[0][0]])
# 		j = j + 1

# img_features_test=sio.loadmat('../../layers /layer22/Flickr_8k.testImages.mat')
# feat_test=img_features_test['features']

# X_test=np.zeros((feat_test.shape[0],feat_test[0,0].shape[2]))
# for i in range(0,feat_test.shape[0]-1) :
#     for j in range(0,feat_test[0,0].shape[2]-2) : 
#         X_test[i,j]=feat_test[i,0][0,0,j]

# ldaf_test = {}
# names = img_features_test["names"]

# sample_size = X_test.shape[0]
# all_scores = np.zeros((names.shape[0], names.shape[0]))

# for k in range(0, sample_size):
# 	j = 0
# 	for name in names:
# 		ldaf_test[name[0][0]] = T[name[0][0]]
# 		all_scores[k][j] = my_cca.score(X_test[k], T[name[0][0]])
# 		j = j + 1

#concatenated projection :
# P=XX*Wx*D

#projected visual features : 
#P_visual=[P[:,1],P[:,2],...P[:,number_of_parameters_for_visual_feature]

#projected tag features :
#P_tag=[P[:,1],P[:,2],...P[:,number_of_parameters_for_tag_feature]

#then we can use CCA.NN(X,tag) to retrieve an image corresponding to a particular tag
