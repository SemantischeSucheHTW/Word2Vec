from indexdao import IndexDao
from pymongo import MongoClient

class Word2VecDao(IndexDao):

    def __init__(self, config):

        '''
        Setup an instance of OrtsIndexDao.
        Keys in config are: host, port, database, collection
        :param config: dict with keys
        '''
        c_copy =  dict(config)
        db = c_copy.pop('db')
        ortsindex_collection = c_copy.pop('word2vec_collection')
              

        self.client = MongoClient(**c_copy)
        self.db = self.client[db]
        self.ortsindex_collection = self.db[ortsindex_collection]
        
        

    def updateIndex(self,  pagedetails):
        # TODO
	return None

    def getUrlfromKey(self, *searchKey, weight=0.0):
        # TODO
        return None
