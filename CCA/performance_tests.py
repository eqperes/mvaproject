"""
Performance Tests for CCA
"""

import numpy as np 
from CCAbib import CCA2view
import scipy.io as sio
import pickle
from sklearn.preprocessing import scale

class PerfTests(object):

	def __init__(self, path_train_view1, path_test_view1, path_view2):
		train_view1 = sio.loadmat(path_train_view1)
		test_view1 = sio.loadmat(path_test_view1)
		view2 = pickle.load(open(path_view2, "rb"))
		self.train_names = train_view1["names"]
		self.test_names = test_view1["names"]
		[self.train_view1, self.train_view2] =\
			self._process(train_view1, view2)
		[self.test_view1, self.test_view2] =\
			self._process(test_view1, view2)

	def _process(self, set_view1, view2):
		view1 = set_view1["features"]
		features1=np.zeros((view1.shape[0],view1[0,0].shape[2]))
		for i in range(0,view1.shape[0]-1) :
		    for j in range(0,view1[0,0].shape[2]-2) : 
		        features1[i,j]=view1[i,0][0,0,j]
		features2 = []
		names = set_view1["names"]
		for name in names:
			features2.append(view2[name[0][0]])
		features2 = np.array(features2)

		return [features1, features2]

	def CCAfy(self, out_dim, normalize=True):
		self.cca = CCA2view(out_dim, normalize)
		self.cca.fit(self.train_view1, self.train_view2)

	def score_matrix(self):
		nsamples = self.test_view1.shape[0]
		scores_matrix = np.zeros((nsamples, nsamples))
		for i in range(0,nsamples):
			dummy_view1 = self.test_view1[i]
			dummy_view1 = np.tile(dummy_view1, (nsamples, 1))
			scores_matrix[i] = self.cca.matching_score(\
				dummy_view1, self.test_view2)
		scores_matrix = scale(scores_matrix, axis=1)
		return scores_matrix

	def score_test(self, out_dim=200, rerun=False, normalize=True):
		if rerun:
			self.CCAfy(out_dim, normalize)
		scores_matrix = self.score_matrix()
		matching_score = np.mean(np.diag(scores_matrix))
		random_score = np.mean(np.mean(scores_matrix))
		return [matching_score, random_score]

	def rank_test(self, out_dim=200, rerun=False, normalize=True):
		if rerun:
			self.CCAfy(out_dim, normalize)
		scores_matrix = self.score_matrix()
		argrank = np.argsort(scores_matrix, axis=1)
		rank = np.argsort(argrank, axis=1)
		return np.mean(np.diag(rank))











		
