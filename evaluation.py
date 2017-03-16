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
			'rec.sport.baseball':['baseball','hit','runs','bat'],
			'rec.sport.hockey':['hockey','period','nhl','ice','skate','goal'],
			'sci.crpyt':['key','encryption','government','chip','clipper','security','privacy','information'],
			'sci.electronics':['power','circuit','ground','wire','current'],
			'sci.space':['space','nasa','earth','launch','orbit','shuttle','moon','solar'],
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


def evaluate(bc,ic,d,num_seeds):
	n_topics = len(bc+ic)
	n_features = 1000
	n_iter = 500

	df = lda_model.DataFetcher(bc,ic,d)
	imbl_docs = df.get_data()
	seed_words = get_topic_seeds(bc+ic,num_seeds)

	n_model = lda_model.LDA_Model(imbl_docs)
	t0 = time()
	n_model.documents_to_topic_model(n_topics,n_features,n_iter,seed_words)
	logger.info("New lda done in {}".format(time()-t0))
	logger.info("New lda perplexity = {}".format("TODO"))
	n_model.display_topics(50)


	df.set_params(bc=bc+ic,ic=None,d=0)
	bal_docs = df.get_data()

	o_model = lda_model.LDA_Model(bal_docs)
	t0 = time()
	o_model.documents_to_topic_model(n_topics,n_features,n_iter,original=True)
	logger.info("Old lda done in {}".format(time()-t0))
	logger.info("Old lda perplexity = {}".format("TODO"))
	o_model.display_topics(50)

	p, r = precision_recall(n_model,o_model)

def precision_recall(n_model,o_model):
	return (1,2)