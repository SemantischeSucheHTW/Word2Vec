from gensim.models import Word2Vec
import os

class SimilarWords:
    
    def __init__(self, config):
        self.model_name = config.pop('model_name')
        self.cwd = os.getcwd()
        self.model = Word2Vec.load(self.cwd + '/pretrained-models/' + self.model_name)
    
    
    def get_similar_words(self, words, topn = 5, drop = 0.4):
        result_with_drop = []
        result_without_drop = []
        for word in words:
            if self.model.wv.vocab.get(word, 0) is not 0:
                # get all words with the 5 highest cosine similarity
                temp = self.model.wv.most_similar(positive=[word], topn=topn)
                result_without_drop.append(temp)
                # remove every word with a lower cosine similarity than drop
                temp_droped=[(y,x) for (y,x) in temp if (x>=drop)]
                result_with_drop.append(temp_droped)                 
               
        return result_with_drop, result_without_drop 
