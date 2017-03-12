import lda_model

if __name__ == '__main__':
	all_seed_words = {'alt.atheism':['god','people','religion','think','question','jesus','evidence','atheism'],
				'comp.graphics':['image','graphics','jpeg','edu','file','images','data','ftp'],
				'comp.os.ms-windows.misc':['ax','max','windows','g9v','b8f','a86','pl','145'],
				'comp.sys.ibm.pc.hardware':['drive', 'scsi','card','controller','disk','ide','pc','bios'],
				'comp.sys.mac.hardware':['mac','apple','drive','problem','software','disk','problem','monitor'],
				'comp.windows.x':['window','file','server','program','motif','widget','x11','version'],
				'misc.forsale':['00','new','sale','10','offer','shipping','price','condition'],
				'rec.autos':['car','cars','engine','new','good','time','oil','speed'],
				'rec.motorcycles':['bike','ride','good','time','motorcycle','new','bikes','dog'],
				'rec.sport.baseball':['baseball','hit','runs','mlb',''],
				'rec.sport.hockey':['hockey','period','nhl','ice','skate','goal'],
				'sci.crpyt':['key','encryption','government','chip','clipper','security','privacy','information'],
				'sci.electronics':['power','circuit','ground','wire','current'],
				'sci.space':['space','nasa','earth','launch','orbit','shuttle','moon','solar'],
				'soc.religion.christian':['god','jesus','church','christ','bible','faith','lord','sin'],
				'talk.politics.guns':['gun','guns','right','fbi','law','firearms','weapons','government'],
				'talk.politics.mideast':['armenian','isreal','turkish','jews','israeli','arab','turkey','war'],
				'talk.politics.misc':['president','people','government','stephanopoulos','mr','right'],
				'talk.religion.misc':['church','religion','homosexual','sex','bible']}
	n_features = 1000
	n_topics = 2
	n_top_words = 20
	n_iter = 1000

	# create an imbalanced set of documents
	# try varying number of balanced and imbalanced categories
	# also try varying d

	### ex 1
	# bc = ['alt.atheism']
	# ic = ['sci.space']
	# d = .9

	### ex 2
	bc = ['alt.atheism','rec.sport.baseball','talks.politics.guns']
	ic = ['sci.space']
	d = .7

	### initialize DataFetcher and get the documents
	df = lda_model.DataFetcher(bc,ic,d)

	### create a balanced set of documents
	# bc = ['alt.atheism', 'sci.space']
	# df = lda_model.DataFetcher(bc)

	documents = df.get_data()

	### set some seed words, refer to all_seed_words
	# seed_words = [['people','god','think'],['space','nasa','earth','launch']]
	seed_words = [['people','god'],['space','nasa']]

	### create the model based on the documents
	model = lda_model.LDA_Model(documents)

	### create the topic model, can compare result using seed words and without seedwords
	# model.documents_to_topic_model(n_topics,n_features) # using no seed words
	model.documents_to_topic_model(n_topics,n_features,n_iter,seed_words) # using seed words
	
	model.display_topics(n_top_words)



