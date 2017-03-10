import lda_model

if __name__ == '__main__':

	n_features = 1000
	n_topics = 2
	n_top_words = 10

	# create an imbalanced set of documents
	bc = ['alt.atheism']
	ic = ['sci.space']
	d = .9
	df = lda_model.DataFetcher(bc,ic,d)
	documents = df.get_data()

	# create a balanced set of documents
	# bc = ['alt.atheism', 'sci.space']
	# documents = get_data(bc)

	seed_words = [['people','god','think'],['space','nasa','earth','launch']]
	model = lda_model.LDA_Model(documents)
	# model.documents_to_topic_model(n_topics,n_features) # using no seed words
	model.documents_to_topic_model(n_topics,n_features,seed_words) # using seed words
	model.display_topics(n_top_words)



