import lda_model
import evaluation
import logging
from time import time
logger = logging.getLogger(__name__)


if __name__ == '__main__':
 	logging.basicConfig(  level=logging.DEBUG)
	### set parameters. n_features is the size of the vocab
	n_features = 1000
	n_topics = 4
	n_top_words = 15
	n_iter = 500

	"""
	imbalanced dataset #1
	"""
	# bc = ['alt.atheism','rec.sport.baseball','talk.politics.guns']
	# ic = ['sci.space']
	bc = ['alt.atheism']
	ic = ['sci.space']
	d = .9
	# for x in range(5):
	# 	evaluation.evaluate(bc,ic,d,num_seeds=x)

	# e = evaluation.evaluate2(bc,ic,d,num_seeds=5)

	# d_array = [.95,.8,.6,.4,.2]
	# for x in range(len(d_array)):
	# 	evaluation.evaluate(bc,ic,d_array[x],num_seeds=5)

	"""
	imbalanced dataset #2
	"""
	bc = ['alt.atheism','rec.sport.baseball']
	ic = ['sci.space']
	d = .8

	# evaluation.evaluate(bc,ic,d,num_seeds=5)
	"""
	imbalanced dataset #3
	"""
	bc = ['alt.atheism','rec.sport.baseball', 'talk.politics.guns']
	ic = ['sci.space']
	d = .8

	# evaluation.evaluate2(bc,ic,d,num_seeds=5)
	# evaluation.evaluate(bc,ic,d,num_seeds=5)

	"""
	imbalanced dataset #4
	"""
	bc = ['alt.atheism','rec.sport.baseball', 'talk.politics.guns','comp.graphics']
	ic = ['sci.space']
	d = .8

	# e = evaluation.evaluate_absolute_pr(bc,ic,d,num_seeds=5,m=5000,n=100)
	# evaluation.evaluate(bc,ic,d,num_seeds=5,m=10,p=.002)

	"""
	imbalanced dataset #5
	"""
	bc = ['alt.atheism','rec.sport.baseball', 'talk.politics.guns','comp.graphics','talk.politics.misc']
	ic = ['sci.space']
	d = .8

	# e= evaluation.evaluate(bc,ic,d,num_seeds=5)
	




