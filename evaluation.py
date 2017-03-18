import lda_model
import logging
from time import time
logger = logging.getLogger(__name__)

ALL_SEED_WORDS = {'alt.atheism':['god','evidence','religion','think','question','jesus','atheism'],
			'comp.graphics':['image','graphics','jpeg','edu','file','images','data','ftp'],
			'comp.os.ms-windows.misc':['ax','max','windows','g9v','b8f','a86','pl','145'],
			'comp.sys.ibm.pc.hardware':['drive', 'scsi','card','controller','disk','ide','pc','bios'],
			'comp.sys.mac.hardware':['mac','apple','drive','problem','software','disk','problem','monitor'],
			'comp.windows.x':['window','file','server','program','motif','widget','x11','version'],
			'misc.forsale':['00','new','sale','10','offer','shipping','price','condition'],
			'rec.autos':['car','cars','engine','new','good','time','oil','speed'],
			'rec.motorcycles':['bike','ride','good','time','motorcycle','new','bikes','dog'],
			'rec.sport.baseball':['baseball','hit','runs','game','player'],
			'rec.sport.hockey':['hockey','period','nhl','ice','skate','goal'],
			'sci.crpyt':['key','encryption','government','chip','clipper','security','privacy','information'],
			'sci.electronics':['power','circuit','ground','wire','current'],
			'sci.space':['space','nasa','earth','launch','shuttle','orbit','moon','solar'],
			'soc.religion.christian':['god','jesus','church','christ','bible','faith','lord','sin'],
			'talk.politics.guns':['gun','guns','right','fbi','law','firearms','weapons','government'],
			'talk.politics.mideast':['armenian','isreal','turkish','jews','israeli','arab','turkey','war'],
			'talk.politics.misc':['president','people','government','stephanopoulos','mr','right'],
			'talk.religion.misc':['church','religion','homosexual','sex','bible']}

def get_topic_seeds(categories,number):
	seeds = []
	for c in categories:
		s = ALL_SEED_WORDS[c][:number]
		seeds.append(s)
	return seeds


def evaluate(bc,ic,d,num_seeds,m=10,p=.002):
	"""
	displays results for modified lda using imbalanced data and original lda using balanced data.
	calculate precision and recall based off words the a probability above a threshold
	"""
	n_topics = len(bc+ic)
	n_features = 1000
	n_iter = 300

	df = lda_model.DataFetcher(bc,ic,d)
	imbl_docs = df.get_data()
	seed_words = get_topic_seeds(bc+ic,num_seeds)

	n_model = lda_model.LDA_Model(imbl_docs)
	t0 = time()
	n_model.documents_to_topic_model(n_topics,n_features,n_iter,seed_words,m=m)
	logger.info("New lda done in {}".format(time()-t0))
	logger.info("New lda perplexity = {}".format(n_model.model.log_perplexity(n_model.X)))
	n_model.display_topics(15)


	df.set_params(bc=bc+ic,ic=None,d=0)
	bal_docs = df.get_data()

	o_model = lda_model.LDA_Model(bal_docs)
	t0 = time()
	o_model.documents_to_topic_model(n_topics,n_features,n_iter,original=True,m=m)
	logger.info("Old lda done in {}".format(time()-t0))
	logger.info("Old lda perplexity = {}".format(o_model.model.log_perplexity(o_model.X)))
	o_model.display_topics(15)

	I = len(bc)
	prs = precision_recall(n_model.get_top_words(p)[I:],o_model.get_top_words(p))
	for pr in prs:
		logger.info("Precision: {}, Recall: {}".format(pr[0],pr[1]))

	return (n_model,o_model)

def evaluate_absolute_pr(bc,ic,d,num_seeds,m=10,n=100):
	"""
	displays results for modified lda using imbalanced data and original lda using balanced data.
	precision/recall calculated based off the top n words instead of all words with a probability above
	a threshold
	"""
	n_topics = len(bc+ic)
	n_features = 1000
	n_iter = 300

	df = lda_model.DataFetcher(bc,ic,d)
	imbl_docs = df.get_data()
	seed_words = get_topic_seeds(bc+ic,num_seeds)

	n_model = lda_model.LDA_Model(imbl_docs)
	t0 = time()
	n_model.documents_to_topic_model(n_topics,n_features,n_iter,seed_words,m=m)
	logger.info("New lda done in {}".format(time()-t0))
	logger.info("New lda perplexity = {}".format(n_model.model.log_perplexity()))
	n_model.display_topics(15)


	df.set_params(bc=bc+ic,ic=None,d=0)
	bal_docs = df.get_data()

	o_model = lda_model.LDA_Model(bal_docs)
	t0 = time()
	o_model.documents_to_topic_model(n_topics,n_features,n_iter,original=True,m=m)
	logger.info("Old lda done in {}".format(time()-t0))
	logger.info("Old lda perplexity = {}".format(o_model.model.log_perplexity()))
	o_model.display_topics(15)

	I = len(bc)
	prs = precision_recall(n_model.get_top_words_absolute(n)[I:],o_model.get_top_words_absolute(n))
	for pr in prs:
		logger.info("Precision: {}, Recall: {}".format(pr[0],pr[1]))

	return (n_model,o_model)

def evaluate2(bc,ic,d,num_seeds):
	"""
	displays the topic words based on the new and old model both using the imbalanced data
	"""
	n_topics = len(bc+ic)
	n_features = 1000
	n_iter = 300

	df = lda_model.DataFetcher(bc,ic,d)
	imbl_docs = df.get_data()
	seed_words = get_topic_seeds(bc+ic,num_seeds)

	n_model = lda_model.LDA_Model(imbl_docs)
	n_model.documents_to_topic_model(n_topics,n_features,n_iter,seed_words)
	n_model.display_topics(20)

	o_model = lda_model.LDA_Model(imbl_docs)
	o_model.documents_to_topic_model(n_topics,n_features,n_iter,original=True)
	o_model.display_topics(20)

	return (n_model,o_model)

def precision_recall(n_topics,o_topics):
	o_topics_ = list(o_topics)
	prs = []
	# compare each new topic to the remaining old topics
	for k,nt in enumerate(n_topics):
		i = 0
		cur = [0,0]
		# find the old topic which gives the highest precision + recall. indicate that topic by j
		# record that precision and recall for the new topic
		for j,ot in enumerate(o_topics_):
			p,r = pr(nt,ot)
			if p+r > cur[0]+cur[1]:
				cur = p,r
				i = j

		# remove topic j for old topics
		logger.debug('Matched Topic {} with Topic {}'.format(len(o_topics)-len(n_topics)+k,i))
		prs.append(cur)
		o_topics_.pop(i)
	return prs

def pr(predicted,expected):
	correct = 0;
	for w in predicted:
		if w in expected:
			correct +=1
	p = correct/float(len(predicted))
	r = correct/float(len(expected))
	return (p,r)