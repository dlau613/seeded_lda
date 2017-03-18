Modified the lda package that can be found [here](https://pypi.python.org/pypi/lda)

Instructions To Run:
Install the lda package with pip install lda. The data needed for our program will be downloaded
automatically.

Run python example.py. This will run the our modified LDA and original LDA on different sets
of balanced and imbalanced data. The modified LDA will have additional seed words as input.
It will print out information including the run time, perplexity, precision and recall, and it
will display some of the top words for each topic. 

Classes:

Created a Model class to wrap the LDA model. Initialize the Model class with a set of documents then call
the documents_to_topic_model method to create the LDA model. display_topics can then be used to display the 
top words for each topic.

Also created a DataFetcher class that will fetch balanced and imbalanced sets of data of specified groups from
the 20 News Group Collection.

Usage:

First get the set of documents you want to create a model on by creating a DataFetcher object then use
the get_data() method. This can grab all the data from the 20 News Group collection, or specific 
documents from specific categories. It can also get an imbalanced set of specified topics.

Next, create an LDA_Model and initialize it with the documents. Now supply it with seed words and other 
parameters to create the topic model.

Finally, you can display the top words in each topic.


Example of Using the DataFetcher and LDA_Model Classes:

import lda_model
import evaluation
import logging
from time import time
logger = logging.getLogger(__name__)

if __name__ == '__main__':
	logging.basicConfig(  level=logging.DEBUG)

	bc = ['alt.atheism','rec.sport.baseball','talk.politics.guns']
	ic = ['sci.space']
	d = .8
	seed_words = [['god','evidence','religion','think','question'], ['baseball','hit','runs','game','player'],
		['gun','guns','right','fbi','law'],['space','nasa','earth','launch','shuttle']]
	df = lda_model.DataFetcher(bc,ic,d)
	documents = df.get_data()

	model = lda_model.LDA_Model(documents)
	model.documents_to_topics(n_topics=4,n_features=1000,n_iter=200,seed_words)
	model.display_topics(15)

