"""
Performance Tests for CCA
"""

import numpy as np 
from CCAbib import CCA2view
import scipy.io as sio
import pickle

class PerfTests(object):

	def __init__(self, path_train_view1, path_test_view1, path_view2, out_dim=200):
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
		features1=np.zeros((set_view1.shape[0],set_view1[0,0].shape[2]))
		for i in range(0,set_view1.shape[0]-1) :
		    for j in range(0,set_view1[0,0].shape[2]-2) : 
		        features1[i,j]=set_view1[i,0][0,0,j]
		features2 = []
		names = set_view1["names"]
		for name in names:
			features2.append(view2[name[0][0]])
		features2 = np.array(features2)

		return [features1, features2]

	def CCAfy(self, out_dim):
		self.cca = CCA2view(out_dim)
		self.cca.fit(train_view1, train_view2)




		
