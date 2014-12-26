"""
Wrapper of CCA as an object 
"""

import numpy as np
from CCA import CCA2
from sklearn.preprocessing import StandardScaler

class CCA2view(object):
	""" Class to manipulate the CCA parameters 
	for the two-view implementation """

	def __init__(self, output_dimension, normalize=True):
		self.out_dim = output_dimension
		self.normalize = normalize

	def fit(self, view1data, view2data):
		if self.normalize:
			self.view_scaler = [StandardScaler(), StandardScaler()]
			view1data = self.view_scaler[0].fit_transform(view1data)
			view2data = self.view_scaler[1].fit_transform(view2data)
		self.view_dimensions = [0, view1data.shape[1], view2data.shape[1]]
		self.view_dimensions = np.cumsum(self.view_dimensions)
		[Wx,D] = CCA2(view1data, view2data)
		self.complete_w = Wx
		self.D = D

	def view_projection(self, view_data, view=1):
		projection_matrix = self.complete_w[\
			self.view_dimensions[view-1]:self.view_dimensions[view],\
			:self.out_dim]
		if self.normalize:
			view_data = self.view_scaler[view-1].transform(view_data)
		return np.dot(view_data, projection_matrix)

	def matching_score(self, view1data, view2data):
		proj1 = self.view_projection(view1data, 1)
		proj2 = self.view_projection(view2data, 2)
		return np.linalg.norm(proj1-proj2, axis=1)




