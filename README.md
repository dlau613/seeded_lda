Modified the lda package that can be found [here](https://pypi.python.org/pypi/lda)

NEEDS TO BE UPDATED: (current example at bottom)
The lda package can quickly be installed with pip install lda.

Everything will work as before except that there is a new method called LDA.fit_seeded(X,seeds) which expects an extra parameter called seeds. Seeds is a list of lists. Each inner list contains indices referring to words in the vocab. These words will be the seeds for a topic.

Created a Model class to wrap the LDA model. Just initialize the Model class with a set of documents then call
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

TO RUN EVALUATIOn

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

	bc = ['alt.atheism','rec.sport.baseball','talk.politics.guns']
	ic = ['sci.space']
	d = .8



