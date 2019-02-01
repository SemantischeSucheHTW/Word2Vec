from indexdao import IndexDao
from pymongo import MongoClient
import os

class Word2VecDao(IndexDao):

    def __init__(self, config):

        '''
        Setup an instance of OrtsIndexDao.
        Keys in config are: host, port, database, collection, model_name
        :param config: dict with keys
        '''
        c_copy =  dict(config)
        db = c_copy.pop('db')
        word2vec_collection = c_copy.pop('word2vec_collection')
        

        self.client = MongoClient(**c_copy)
        self.db = self.client[db]
        self.word2vec_collection = self.db[word2vec_collection]
        
        

    def updateIndex(self,  model_name ):
        cwd = os.getcwd()
        model = Word2Vec.load(cwd + '/pretrained-models/' + model_name)
        
        for key in model.wv.vocab.keys():
            self.word2vec_collection.update_one(
                    {'word': key},
                    {'$set': { "word_embedding" : model.wv.word_vec(key)}},
                    upsert=True)
        return None

    def getUrlfromKey(self, *searchKey, weight=0.0):
        urls = []
        for key in searchKey:
            result = self.word2vec_collection.find({'word' : key.lower()})
            for doc in result:
                for url in doc["urls"]:
                    urls.append(url)
        return (urls, weight)
