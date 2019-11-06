# local imports
from DTM import utils

# Library imports
import numpy as np

# built-in imports
import sys

class DTM():
    def __init__(self):
        '''
        '''
        self.vocab = None
        self.DTM = None
        self.word_frequencies = dict()

        # hidden variables
        self._vocab_idx_map = dict()
        self._doc_count = 0


    def build(self, sentences):
        self._build_vocab(sentences)
        self.DTM = self._build_DTM(sentences)


    def _build_vocab(self, sentences):
        '''Build The vocab array'''
        vocab = set()

        for tokens in sentences:
            self._doc_count += 1

            for token in tokens:
                vocab.add(token)
                if token not in self.word_frequencies:
                    self.word_frequencies[token] = 0
                self.word_frequencies[token] += 1

        self.vocab = sorted(vocab)
        self._vocab_idx_map = {w:ix for ix,w in enumerate(self.vocab)}


    def _build_DTM(self, sentences):
        '''Build the Term-Document matrix'''
        docs = np.zeros((self._doc_count,len(self.vocab)))

        for doc_count, tokens in enumerate(sentences):

            for token in tokens:
                ix = self._vocab_idx_map[token]
                docs[doc_count, ix] = tokens.count(token)

        return docs


    def word_2_word_sim(self, w1, w2):
        '''
            parameters:
                w1 (string): first word
                w2 (string): second word

            returns:
                float : cosine similarity of the two words
        '''
        try:
            w1_i = utils.binary_search(self.vocab, w1)
            w1_i = self._vocab_idx_map[w1]
            w2_i = self._vocab_idx_map[w2]
        except:
            # The word isn't in the vocab
            return 0

        w1_vec = self.DTM[:, w1_i]
        w2_vec = self.DTM[:, w2_i]
        sim = utils.cosine_sim(w1_vec, w2_vec)
        return sim


    def calculate_all_word_sims(self, cutoff=10, tol=0.0):
        '''
            parameters:
                cuttoff (int): word frequency cutoff. Calcualting the pairwise cosine similairty of 
                    every word in the vocab may be computationally infeasible
                tol (float): any word-pair similarities below this threshold will not be returned
            return:
                list: list of triples containing two words and their similarity
        '''

        # find all indicies above the cutoff
        ixs = [self._vocab_idx_map[token] for token,freq in self.word_frequencies.items() if freq >= cutoff]
        all_cossims = utils.pairwise_cossim(self.DTM[:, ixs].T)

        similarities = []

        N,M = all_cossims.shape
        for i in range(N):
            for j in range(M):
                if i==j:
                    continue

                sim = all_cossims[i,j]
                if sim<=tol:
                    continue

                v_i = ixs[i]
                v_j = ixs[j]
                similarities.append( (self.vocab[v_i], self.vocab[v_j], sim) )
        return similarities