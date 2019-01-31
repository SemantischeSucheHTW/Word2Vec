import numpy as np
import re
from tqdm import tqdm 
import time

def clean_document(raw_documents):
    '''
    this method gets a list of documents and removes all non alphabetic characters and
    except for ü, ä, ö, ß
    :param raw_documents: a list of documents e.g. [ "I 12hallo bed", "1\sleep;"]
    :return: a list of cleaned documents e.g. ["I hallo bed", I sleep"]
    '''
    print("starting cleaning texts")
    regex = re.compile('[^a-z+ü+ä+ö+ß]')
    cleaned = [regex.sub(' ', str(document).lower()) for document in raw_documents]
    return cleaned


def flatten(list):
    '''
    this method flattens out a 2d list
    :param lists: 2d list e.g. [[1,2],[3,4]]
    :return: 1d list e.g. [1, 2, 3, 4]
    '''
    new_list = []
    for i in list:
        for j in i:
            new_list.append(j)
    return new_list


def get_vocabulary(tokenized_documents):
    '''
    this method gets a list of tokenized documents and returns a list of words which appears in the whole document
    alias our vocabulary
    :param tokenized_documents: e.g. [ [ "I", "am", "sleep", "over", "I"], [ "hello", "word", "sleep"]
    :return: the vocabulary e.g. ["I", "am", "sleep", "over", "hello", word"]
    '''
    flattend = flatten(tokenized_documents)
    vocabulary = list(set(flattend))
    return vocabulary


def tokenize_documents(document):
    '''
    this method gets a list of documents and returns the documents in a tokenized form
    :param document: a list of documents e.g. [["I hallo bed"], ["I sleep"]]
    :return: tokenized documents e.g. [ ["I", "hallo", "bed"], ["I", "sleep"]]
    '''
    print("starting tokenizing")
    tokenized = [text.split(" ") for text in document]
    tokenized = [[x for x in tokens if x] for tokens in tokenized]
    return tokenized


def extract_window(documents, window_size=3):
    """
    deprecated
    """
    print('deprecated')
    center_words = []
    context_words = []
    print("starting extracting windows")
    for doc in tqdm(documents):
        context_word = np.array([doc[i:i + window_size]
                                 for i in range(len(doc) - window_size - 1)])
        center_word = np.array([context_word[i][int(window_size / 2)]
                                for i in range(len(context_word))])
        context_words.append(context_word)
        center_words.append(center_word)
    return np.array(context_words), np.array(center_words)


def document_to_onehot_encoding(tokenized, words):
    """
    deprecated
    """
    print('deprecated')
    onehot_documents = []
    print("starting onehot encoding")
    onehot_map = {}
    for i, word in tqdm(enumerate(words)):
        onehot_map[word] = np.zeros((len(words),1))
        onehot_map[word][i] = 1
    
    for token in tqdm(tokenized):
        onehot = np.array([onehot_map[tk] for tk in token])
        onehot_documents.append(onehot)
    return np.array(onehot_documents)


def _create_trainings_dataset(raw_texts, window_size=3):
    '''
    deprecated
    '''
    t0 = time.time()
    cleaned_texts = clean_document(raw_texts)
    tokenized = tokenize_documents(cleaned_texts)
    words = get_vocabulary(tokenized)
    onehot = document_to_onehot_encoding(tokenized, words)
    return onehot
    context, center = extract_window(onehot, window_size=window_size)
    t1 = time.time()
    print("dataset created", t1 - t0, "ms")
    return words, context, center
