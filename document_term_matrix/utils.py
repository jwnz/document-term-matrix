import numpy as np

class DocumentIterator():
    def __init__(self, input_file, delim='\t'):
        self.input_file = input_file
        self.delim = delim

    def __iter__(self):
        for line in open(self.input_file, encoding='utf-8'):
            yield line[:-1].split(self.delim)


def binary_search(lst, keyword):
    l = 0
    r = len(lst)-1
    while l <= r:
        m = (l+r)//2
        if lst[m] < keyword:
            l = m+1
        elif lst[m] > keyword:
            r = m-1
        else:
            return m
    return -1

def cosine_sim(A,B):
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot / (norma * normb)
    return cos

def pairwise_cossim(A):
    # https://stackoverflow.com/a/20687984


    # base similarity matrix (all dot products)
    # replace this with A.dot(A.T).toarray() for sparse representation
    similarity = np.dot(A, A.T)


    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine