from gensim.models import Word2Vec
import os
class SimilarWords:
    
    def __init__(self, config):
        #self.model_name = config.pop('model_name')
        #self.cwd = os.getcwd()
        #self.model = Word2Vec.load(self.cwd + '/pretrained-models/' + self.model_name)
        
    def _get_similar_words(self, words):
        # TODO
    
    def _get_positive_similar_words(self, word, topn = 5):
        """
        deprecated
        """
        print('deprecated')
        if self.model.wv.vocab.get(word, 0) is not 0:
            return self.model.wv.most_similar_cosmul(positive=word, topn=topn)
    
    def _get_negative_similar_words(self, word, topn = 5):
        """
        deprecated
        """
        print('deprecated')
        if self.model.wv.vocab.get(word, 0) is not 0:
            return self.model.wv.most_similar_cosmul(positive=word, negative=word, topn=topn)
