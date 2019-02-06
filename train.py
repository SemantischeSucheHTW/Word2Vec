from __future__ import print_function
import os
import sys
import argparse

import pandas as pd
import numpy as np
from gensim.models import Word2Vec

from model.preprocessor import *

pretrained_dir = 'pretrained-models/'
data_dir = 'data/'

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--outfile', help='output file in which the Word2Vec Model will be saved (every model will be saved at the pretrained-models directory) - default name is word2vec.model',
                        default='word2vec.model')
    parser.add_argument('-d', '--data', help='dataset on which the word2vec model will be trained on (must be a CSV with the label text and in the data directory)  - default is the police report datasetl',
                        default='police_reports.csv')
    parser.add_argument('-wk', '--worker', help='how many workes will be used ',type=int,
                        default=1)
    parser.add_argument('-wi', '--window', help='maximum distance between the current and predicted word within a sentence',type=int, required=True)
    parser.add_argument('-s', '--size', help='dimensionality of the word vectors.',type=int,
                        default=100)
    parser.add_argument('-m', '--mincount', help='how many words will be ignored with the total frequency lower than this number',type=int,
                        default=1)
    parser.add_argument('-e', '--epoch', help='number of iterations over the corpus - how often our model will see our dataset',type=int,
                        default=100)
    args = parser.parse_args(arguments)
    print(args)
    # data preparation
    df = pd.read_csv(data_dir + args.data)
    raw_texts = df.text.get_values()
    cleaned_texts = clean_document(raw_texts)
    tokenized = tokenize_documents(cleaned_texts)
    words = get_vocabulary(tokenized)
    # actuall training phase
    print('started training')
    model = Word2Vec(tokenized, size=args.size, window=args.window, min_count=args.mincount, workers=args.worker)
    model.train(tokenized, total_examples=model.corpus_count, epochs=args.epoch)
    print('training ended - model will be safed at', pretrained_dir + args.outfile)
    model.save(pretrained_dir + args.outfile)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
