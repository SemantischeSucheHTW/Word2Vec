from __future__ import print_function
import os
import sys
import argparse


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--outfile', help='output file in which the Word2Vec Model will be saved - default name is word2vec.model',
                        default='word2vec.model')
    parser.add_argument('-w', '--workers', help='how many workes will be used ',
                        default=1)
    parser.add_argument('-s', '--size', help='dimensionality of the word vectors.',
                        default=100)
    parser.add_argument('-m', '--mincount', help='how many words will be ignored with the total frequency lower than this number',
                        default=1)
    parser.add_argument('-e', '--epoch', help='number of iterations over the corpus - how often our model will see our dataset',
                        default=100)

    args = parser.parse_args(arguments)
    print(args)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
