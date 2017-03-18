from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

from stopwords import STOP_WORDS
import numpy as np
import mod_lda


import logging

logger = logging.getLogger(__name__)

class DataFetcher:
	def __init__(self,balanced_categories=None,imbalanced_categories=None,discard=0):
		self.balanced_categories = balanced_categories
		self.imbalanced_categories = imbalanced_categories
		self.discard = discard
		if len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.NullHandler):
			logging.basicConfig(level=logging.DEBUG)

	def set_params(self,bc=None,ic=None,d=0):
		self.balanced_categories = bc
		self.imbalanced_categories = ic
		self.discard = d	
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

		it return a list of text documents
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
	def __init__(self, documents=None,max_df=.50,min_df=2):
		self.documents = documents
		self.max_df = max_df
		self.min_df = min_df
		if len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.NullHandler):
			logging.basicConfig(level=logging.DEBUG)
	def set_documents(self,documents):
		self.documents = documents

	def documents_to_topic_model(self,n_topics,n_features,n_iter,seed_words=None,original=False,m=10):
		print("Extracting tf features for LDA...")
		# We use a few heuristics to filter out useless terms early on: the posts are stripped of headers,
		# footers and quoted replies, and common English words, words occurring in only one document or 
		# in at least 95% of the documents are removed. Use tf (raw term count) features for LDA.
		# tf is a csr_matrix 
		tf_vectorizer = CountVectorizer(max_df=self.max_df, min_df=self.min_df, max_features=n_features,stop_words=STOP_WORDS)
		tf = tf_vectorizer.fit_transform(self.documents)
		self.vocab = tf_vectorizer.get_feature_names()
		tf2 = self.remove_zero_rows(tf)
		self.X = X = tf2.toarray()
		
		# TODO: currently crashes if a seed_word is not in the vocab
		seeds = self.get_seed_indices(seed_words)

		self.model = mod_lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=1,refresh=100)
		if original:
			self.model.fit(X)
		else:
			self.model.fit_seeded(X,seeds,m)

	def test(self,n):
		return self.model.transform(self.X[n])


	def remove_zero_rows(self,X):
	    # X is a scipy sparse matrix. We want to remove all zero rows from it
	    nonzero_row_indice, _ = X.nonzero()
	    unique_nonzero_indice = np.unique(nonzero_row_indice)
	    logger.info("Removed {} all zero rows".format(X.shape[0]-X[unique_nonzero_indice].shape[0]))
	    return X[unique_nonzero_indice]

	def get_seed_indices(self,seed_words):
		if seed_words == None:
			return None
		seeds = []
		for i,topic in enumerate(seed_words):
			s = []
			for j,seed in enumerate(topic):
				if seed in self.vocab:
					s.append(self.vocab.index(seed))
				else:
					logger.info("The seed word '{}' wasn't in the vocabulary".format(seed))
			seeds.append(s)
		return seeds

	def display_topics(self, n_top_words):
	    topic_word = self.model.topic_word_
	    for i, topic_dist in enumerate(topic_word):
	    	topic_words = np.array(self.vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	    	print('Topic {}: {}'.format(i, ' '.join(topic_words)))
	    	indices = np.argsort(topic_dist)
	    	# logger.debug(indices)
	    	logger.debug(topic_dist[indices[len(indices)-n_top_words]])

	def get_top_words(self,p=.002):
		topic_word = self.model.topic_word_
		top_words = []
		for i,topic_dist in enumerate(topic_word):
			indices = np.argsort(topic_dist)[::-1]
			I = len(indices)
			for j in range(1,len(indices)):
				if topic_dist[indices[j]] < p:
					I = j
					logger.debug('Topic {} has {} words with p>{}'.format(i,I,p))
					break
			temp = np.array(self.vocab)[indices[:I]]
			# print('Topic {}: {}'.format(i,' '.join(temp)))
			top_words.append(temp)
		return top_words

	def get_top_words_absolute(self,n=100):
		topic_word = self.model.topic_word_
		top_words = []
		for i,topic_dist in enumerate(topic_word):
			topic_words = np.array(self.vocab)[np.argsort(topic_dist)][:-n+1:-1]
			top_words.append(topic_words)
		return top_words
