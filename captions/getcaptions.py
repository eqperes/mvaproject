import pickle
import unittest
import numpy as np
from numpy import genfromtxt

def import_captions(captions_path):
	""" Import the captions for flickr8 images """
	
	captions = genfromtxt(captions_path, dtype="string", delimiter="\t",\
		comments='***')
	captions_dict = {}	
	for row in captions:
		name = row[0].split("#")[0]
		if name not in captions_dict:
			captions_dict[name] = [[word.lower() for \
					word in row[1].split()]]
		else:
			captions_dict[name].append([word.lower() for \
					word in row[1].split()])

	return captions_dict

def subset_captions(subset_path, captions_dict):
	""" Obtain subset selecting only names in the txt file """
	with open(subset_path) as f:
		names = f.read().splitlines()
	subset_dict = {}
	for name in names:
		if name in captions_dict:
			subset_dict[name] = captions_dict[name]
	return subset_dict

def run_tests():
	"""
	Run tests for this module
	"""
	suite = unittest.TestLoader().\
		loadTestsFromTestCase(TestCaptionsFunctions)
	unittest.TextTestRunner(verbosity=2).run(suite)

class TestCaptionsFunctions(unittest.TestCase):	
	""" TestCase for caption functions """
	
	def test_import_captions(self):
		""" Test the importation of captions """
		captions_path = "./test_files/test_captions.txt"
		result_path = "./test_files/test.dict"

		captions = import_captions(captions_path)
		fin = open(result_path, "rb")
		result = pickle.load(fin)
		self.assertEqual(result, captions)

	def test_subset_captions(self):
		""" Test the selection of captions from subset """
		captions_path = "./test_files/test_captions.txt"
		subset_path = "./test_files/subset_test.txt"
		result_path = "./test_files/subset.dict"

		captions = import_captions(captions_path)
		subset = subset_captions(subset_path, captions)
		fin = open(result_path, "rb")
		result = pickle.load(fin)
		self.assertEqual(result, subset)



