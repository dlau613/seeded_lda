from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

import numpy as np
import lda

class DataFetcher:
	def __init__(self,balanced_categories=None,imbalanced_categories=None,discard=0):
		self.balanced_categories = balanced_categories
		self.imbalanced_categories = imbalanced_categories
		self.discard = discard

	def set_balanced_categories(self,balanced_categories):
		self.balanced_categories = balanced_categories

	def set_imbalanced_categories(self,imbalanced_categories):
		self.imbalanced_categories = imbalanced_categories

	def set_discard(self,discard):
		self.discard = discard

	def get_data(self):
		"""
		balanced_categories are the balanced topics. 
		imbalanced_categories are the imbalanced topics. 
		d is the fraction of the documents to discard for the imbalanced topics.
		If no params given then all the data will be fetched.
		If only balanced categories are given then only those will be fetched.
		"""
		if self.imbalanced_categories==None:
			if self.balanced_categories == None:
				dataset = fetch_20newsgroups(subset='all',shuffle=True, random_state=1,remove=('headers','footers','quotes'))
				return dataset.data
			else:
				dataset = fetch_20newsgroups(subset='all',categories = self.balanced_categories,shuffle=True,random_state=1,remove=('headers','footers','quotes'))
				return dataset.data
		else:
			documents = fetch_20newsgroups(subset='all',categories=self.balanced_categories,shuffle=True,random_state=1,remove=('headers','footers','quotes')).data
			for c in self.imbalanced_categories:
				# shuffle=False to get a consistent imbalanced dataset
				docs = fetch_20newsgroups(subset='all',categories=[c],shuffle=False,random_state=1,remove=('headers','footers','quotes')).data
				documents += docs[:int(len(docs)*(1-self.discard))]
			np.random.shuffle(documents)
			return documents



class LDA_Model:
	def __init__(self, documents):
		self.documents = documents

	def documents_to_topic_model(self,n_topics,n_features,seed_words=None,original=False):
		print("Extracting tf features for LDA...")
		# We use a few heuristics to filter out useless terms early on: the posts are stripped of headers,
		# footers and quoted replies, and common English words, words occurring in only one document or 
		# in at least 95% of the documents are removed. Use tf (raw term count) features for LDA.
		# tf is a csr_matrix 
		tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,stop_words='english')
		tf = tf_vectorizer.fit_transform(self.documents)
		self.vocab = tf_vectorizer.get_feature_names()
		tf2 = self.remove_zero_rows(tf)
		X = tf2.toarray()
		
		seeds = self.get_seed_indices(seed_words)

		self.model = lda.LDA(n_topics=n_topics, n_iter=900, random_state=1)
		if original:
			self.model.fit(X)
		else:
			self.model.fit_seeded(X,seeds)

	def remove_zero_rows(self,X):
	    # X is a scipy sparse matrix. We want to remove all zero rows from it
	    nonzero_row_indice, _ = X.nonzero()
	    unique_nonzero_indice = np.unique(nonzero_row_indice)
	    return X[unique_nonzero_indice]

	def get_seed_indices(self,seed_words):
		if seed_words == None:
			return None
		seeds = []
		for i,topic in enumerate(seed_words):
			s = []
			for j,seed in enumerate(topic):
				s.append(self.vocab.index(seed))
			seeds.append(s)
		return seeds

	def display_topics(self, n_top_words):
	    topic_word = self.model.topic_word_
	    for i, topic_dist in enumerate(topic_word):
	    	topic_words = np.array(self.vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	    	print('Topic {}: {}'.format(i, ' '.join(topic_words)))