### Document-term-matrix

This is an incomplete document-term matrix python <i>library</i>.



Initialize the model, load the text data (list of lists of strings), and run the build function.
```python
from document_term_matrix.DocumentTermMatrix import DocumentTermMatrix
from document_term_matrix import utils

dtm = DocumentTermMatrix()
sentences = utils.DocumentIterator('preprocessed_text_data.txt', delim='\t')
dtm.build(sentences)
```

The Document-Term Matrix is a numpy 2D array. The rows represent the documents and the columns represent each individual word in the (sorted) vocab.
```python
dtm.DTM
>>> array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])

# first document
dtm.DTM[0, :]
>>> array([0., 0., 0., ..., 0., 0., 0.])

# vector representing the first word
dtm.DTM[:, 0]
>>> array([0., 0., 0., ..., 0., 0., 0.])
```

Check the word frequencies as follows:
```python
dtm.word_frequencies

>>> { 'linkin': 1,
      'park': 29,
      'in': 199,
      'the': 455,
      'end': 6,
      'it': 162,
      'starts': 1,
      'with': 62,
      'one': 46,
      'thing': 2, 
      ... }
```



Calculate the cosine similarity between any two words in the vocab:
```python
dtm.word_2_word_sim('the', 'with')
>>> 0.18393539930448133
```

Compute the parwise (cosine) similarity for all word pairs in the vocab.<br>
This generally requires a lot of memory, so it's recommended to set the <i>cutoff</i> parameter to filter out words with
low frequencies to reduce the size of the matrix. Additionally, the <i>tolerance</i> parameter can be used to filter out any words pairs with a similarity below the given value:

```python
dtm.calculate_all_word_sims(cutoff=100, tol=0.1)

>>> [('com', 'baby', 0.12184896725606949),
    ('com', 'gmail', 0.18702757081485133),
    ('com', 'run', 0.11082707844047872),
    ('com', 'hotmail', 0.32323264425116566),
    ('com', 'hit', 0.16800419993169763),
    ('com', 'mlb', 0.1407256316981406),
    ('com', 'specmade', 0.1224324170635966),
    ('com', 'gun', 0.12785990088125448),
    ('com', 'specspot', 0.11054585572721631),
    ('com', 'ufc', 0.10677272299600572)]
    ...]
```

<br>
<hr>
<br>

### TF-IDF
The library also has supoprt for basic term frequency and inverse document frequency functionality.

Initialize the `DocumentTermMatrix` obejct as usual. Then run the `build` function to generate the document-term matrix given the specified tf and idf functions.

```python
import document_term_matrix.DocumentTermMatrix as DTM
dtm = dtm.DocumentTermMatrix(tf='freq', idf='idf')
dtm.build(sentences)
```

In the case of using the <i>double normalization K</i> term frequency function, the `norm_k` parameter should also be set. The default value is `0.5`.

```python
dtm = dtm.DocumentTermMatrix(tf='doublenormk', idf='idf', norm_k=0.2)
```

The included term frequency and inverse document frequency functions are as follows:

| TF             |          | IDF           |          |
|----------------|----------|---------------|----------|
| Key            | Function | Key           | Function |
| count          |          | idf           |          |
| binary         |          | smooth        |          |
| freq           |          | max           |          |
| lognorm        |          | probabilistic |          |
| doublenormhalf |          |               |          |
| doublenormk    |          |               |          |


Users may specify a specific function for the term frequency or inverse document frequency as well. The function will be executed on the term-document matrix using the numpy function `apply_along_axis(func, 1, dtm)` and `apply_along_axis(func, 0, dtm)` respectively.

```python
# example of weighting scheme 2 from the wikipedia article
tf = lambda x: 1+np.log(x)
idf = lambda x: np.log(1+(dtm.DTM.shape[0]/np.count_nonzero(x[x>0])))

dtm = dtm.DocumentTermMatrix(tf_func=tf, idf_func=idf)
```