import nltk

SOS = "<s> "
EOS = "</s>"
UNK = "<UNK>"

"""Add Sentence Tokens:
    
    To identify the beginning and end of the sentence 
    add the StartOfSentence and EndOfSentence tokens.

    The argument 'sentences' takes a list of str and 'n' is the order of the model.
    The function returns the list of generated of sentences.
    
    For bigram models (or greater) both tokens are added otherwise or only one is added.
"""

def add_sentence_tokens(sentences, n):
    sos = SOS * (n-1) if n > 1 else SOS
    return ['{}{} {}'.format(sos, s, EOS) for s in sentences]

"""Replace singletons:
    
    For the tokens appearing only ones in the corpus, replace it with <UNK>
    
    The argument 'tokens' takes input of the tokens comprised in the corpus.
    The function returns list of tokens after replacing each singleton with <UNK>

    It is done to tackle the words which are not present in the vocabulary of the corpus (i.e., Out of Vocabulary words). 
    It assignes <UNK> token to the words which appeared only one time. 
    Then if new word is encountered, it is treated, its probility is assigned as the probability of <UNK> token.
"""

def replace_singletons(tokens):
    vocab = nltk.FreqDist(tokens)       
    """FreqDist() returns dictionary with key as tokens and values as corresponding frequency of the token"""
    return [token if vocab[token] > 1 else UNK for token in tokens]

"""Preprocess:
    
    The function takes the argument 'sentences' that takes the list of str of
    preprocess. The argument 'n' is the order of the model.
    Adds the above three tokens to the sentences and tokenize.
    The function returns preprocessed sentences.
"""

def preprocess(sentences, n):
    sentences = add_sentence_tokens(sentences, n)
    tokens = ' '.join(sentences).split(' ')
    tokens = replace_singletons(tokens)
    return tokens



import argparse
from itertools import product
import math
from pathlib import Path

"""    This function loads training and testing corpus from a directory.
    The argument 'data_dir' contains path of the directory. The directory should contain files: 'train.txt' and 'test.txt'
    Function returns train and test sets as lists of sentences.
"""

def load_data(data_dir):
    train_path = data_dir + 'train.txt'
    test_path  = data_dir + 'test.txt'

    with open(train_path, 'r') as f:
        train = [l.strip() for l in f.readlines()]
    with open(test_path, 'r') as f:
        test = [l.strip() for l in f.readlines()]
    return train, test

"""Trained N-gram model:

    A trained model for the given corpus is constructed by preprocessing the 
    corpus and calculating the smoothed probabilities of each n-gram. 
    The arguments contains training data (list of strings), n (integer; order of the model), 
    and an integer used for laplace smoothing.
    Further, the model has a method for calculating perplexity.
"""

class LanguageModel(object):
    def __init__(self, train_data, n, laplace=1):
        self.n = n
        self.laplace = laplace
        self.tokens = preprocess(train_data, n)
        self.vocab  = nltk.FreqDist(self.tokens)
        self.model  = self._create_model()
        self.masks  = list(reversed(list(product((0,1), repeat=n))))

    def _smooth(self):
        """
        The n tokens of n-gram in training corpus and first n-1 tokens of each n-gram
        results in Laplace smoothenedd probability.
        The function returns the smoothened probability mapped to its n-gram.

        """
        vocab_size = len(self.vocab)

        n_grams = nltk.ngrams(self.tokens, self.n)
        n_vocab = nltk.FreqDist(n_grams)

        m_grams = nltk.ngrams(self.tokens, self.n-1)
        m_vocab = nltk.FreqDist(m_grams)

        def smoothed_count(n_gram, n_count):
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            return (n_count + self.laplace) / (m_count + self.laplace * vocab_size)

        return { n_gram: smoothed_count(n_gram, count) for n_gram, count in n_vocab.items() }

    def _create_model(self):
        """
        This function creates a probability distribution of the vocabulary of training corpus.
        The probabilities in a unigram model are simply relative frequencies of each token over the whole corpus.
        Otherwise, the relative frequencies are Laplace-smoothed probabilities.
        Function returns a dictionary which maps each n-gram, which is in the form of tuple of strings, to its probabilities (float)

        """
        if self.n == 1:
            num_tokens = len(self.tokens)
            return { (unigram,): count / num_tokens for unigram, count in self.vocab.items() }
        else:
            return self._smooth()

    def _convert_oov(self, ngram):
        """
        This function handles the words which are encountered in the test and converts the given n-gram to one which is known by the model.
        Stop when the model contains an entry for every permutation.
        The function returns n-gram with <UNK> tokens in certain positions such that the model
            contains an entry for it.
        """
        mask = lambda ngram, bitmask: tuple((token if flag == 1 else "<UNK>" for token,flag in zip(ngram, bitmask)))

        ngram = (ngram,) if type(ngram) is str else ngram
        for possible_known in [mask(ngram, bitmask) for bitmask in self.masks]:
            if possible_known in self.model:
                return possible_known

    def perplexity(self, test_data):
        """
        Perplexity of the model is calculated using the sentences and returns
        a float value. 
        
        """
        test_tokens = preprocess(test_data, self.n)
        test_ngrams = nltk.ngrams(test_tokens, self.n)
        N = len(test_tokens)

        known_ngrams  = (self._convert_oov(ngram) for ngram in test_ngrams)
        probabilities = [self.model[ngram] for ngram in known_ngrams]

        return math.exp((-1/N) * sum(map(math.log, probabilities)))

    def _best_candidate(self, prev, i, without=[]):
        """
        Selects the most probable token depending on the basis of previous
        (n-1) tokens. 
        The function takes the argument of previous (n-1) tokens, and the tokens to
        exclude from candidates list.
        The function returns the most probable token and its probability.

        """
        blacklist  = ["<UNK>"] + without
        candidates = ((ngram[-1],prob) for ngram,prob in self.model.items() if ngram[:-1]==prev)
        candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
        candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        if len(candidates) == 0:
            return ("</s>", 1)
        else:
            return candidates[0 if prev != () and prev[-1] != "<s>" else i]

data_path = '/content/drive/Shareddrives/MathProject22/Dataset/data/'
train, test = load_data(data_path)

#if __name__ == '__main__':
model_instance= LanguageModel(train[0:1000000], 3, 0)
   # first number is the n of n gram
   # second number is the coefficient whether laplace used or not

print(model_instance.perplexity(test))

prev=('I','love',)
print(model_instance._best_candidate(prev,1)[0])
# `1 is ith best fit as a candidate

import pickle
filename = 'without_laplace.sav'
pickle.dump(model_instance, open(filename, 'wb'))

len(train)
