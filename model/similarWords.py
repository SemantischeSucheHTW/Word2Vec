from gensim.models import Word2Vec
import os

class SimilarWords:
    
    def __init__(self, config):
        self.model_name = config.pop('model_name')
        self.cwd = os.getcwd()
        self.model = Word2Vec.load(self.cwd + '/pretrained-models/' + self.model_name)
    
    def get_positive_similar_words(self, words, topn = 5):
        result = []
        for word in words:
            if self.model.wv.vocab.get(word, 0) is not 0:
                result.append(self.model.wv.most_similar(positive=[word], topn=topn))
        return result 
    
    def get_negative_similar_words(self, words, topn = 5):
        result = []
        for word in words:
            if self.model.wv.vocab.get(word, 0) is not 0:
                 result.append(self.model.wv.most_similar(negative=[word], topn=topn))
        return result