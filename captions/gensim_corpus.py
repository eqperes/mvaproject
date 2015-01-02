import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
import numpy as np
from scipy.spatial import distance


class CaptionCorpus(object): 
	""" Class to manipulate the corpus of captions """

	def __init__(self, captions_dict, stop="None"):
		self.captions = captions_dict
		self.documents = []
		for captions in self.captions.values():
			self.documents += captions
		self.stop_words = []
		if stop == "English":
			self.stop_words = stopwords.words('english') + [".", ","]
			self.documents = [self._stop_document(document) \
				for document in self.documents]
		self.dictionary = corpora.Dictionary(self.documents)

	def _stop_document(self, document):
		return [word for word in document if word not in \
			self.stop_words]

	def bow_corpus(self):
		""" Return the bag of words corpus """
		return [self.dictionary.doc2bow(document) for \
			document in self.documents]

	def ldafy(self, num_topics): 
		self.lda = LdaModel(self.bow_corpus(), num_topics=num_topics, \
			id2word=self.dictionary)

	def word2vecfy(self, size, min_count):
		self.w2v = Word2Vec(self.documents, size=size, \
			min_count=min_count)

	def _lda_vector(self, document):
		document = self._stop_document(document)
		document = self.dictionary.doc2bow(document)
		vector = self.lda.inference([document])[0][0]
		return vector

	def lda_corpus(self):
		lda_dict = {}
		for name in self.captions:
			captions = self.captions[name]
			vector = []
			for caption in captions:
				lda_caption = self._lda_vector(caption)
				vector.append(lda_caption)
			lda_dict[name] = np.mean(vector, axis=0)
		return lda_dict

	def w2v_pretrained(self, path_to_model):
		self.w2v = Word2Vec.load_word2vec_format(\
			path_to_model, binary=True)

	def w2v_corpus(self):
		w2v_dict = {}
		for name in self.captions:
			captions = self.captions[name]
			for caption in captions:
				w2v_caption = self._w2v_document(caption)
				if not name in w2v_dict:
					w2v_dict[name] = w2v_caption
				else:
					w2v_dict[name] = w2v_dict[name] + w2v_caption
		return w2v_dict

	def lda_distance(self, document1, document2):
		vector1 = self._lda_vector(document1)
		vector2 = self._lda_vector(document2)
		vector1 = vector1/np.linalg.norm(vector1)
		vector2 = vector2/np.linalg.norm(vector2)
		return np.linalg.norm(vector1-vector2)

	def _w2v_document(self, document):
		document = self._stop_document(document)
		vectors = []
		for word in document:
			try:
				vectors.append(self.w2v[word])
			except:
				pass
		return np.mean(vectors, 0)

	def w2v_distance(self, document1, document2):
		vector1 = self._w2v_document(document1)
		vector2 = self._w2v_document(document2)
		return np.linalg.norm(vector1-vector2)

	def image_features(self, method="lda"): 
		img_features = {}
		for name in self.captions: 
			captions = self.captions[name]
			if method=="lda":
				vectors = [self._lda_vector(doc) \
							for doc in captions]
			else:
				vectors = [self._w2v_document(doc) \
							for doc in captions]
			img_features[name] = np.mean(vectors, 0)
		return img_features

	def w2v_image_caption_distance(self, document):
		img_distances = {}
		for name in self.captions: 
			captions = self.captions[name]
			dists = np.zeros(len(captions))
			for i in range(0, len(captions)):
				dists[i] = self.w2v_distance(document, captions[i])
			img_distances[name] = np.mean(dists)
		return img_distances

	def lda_image_caption_distance(self, document):
		img_distances = {}
		for name in self.captions: 
			captions = self.captions[name]
			dists = np.zeros(len(captions))
			for i in range(0, len(captions)):
				dists[i] = self.lda_distance(document, captions[i])
			img_distances[name] = np.mean(dists)
		return img_distances



