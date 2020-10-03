### Document-term-matrix

This is a document-term matrix python library for small tasks that fit in memory.

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
low frequencies to reduce the size of the matrix. Additionally, the <i>tolerance</i> parameter can be used to filter out any words pairs with a similarity below the given value. <small>(Note: The values returned are not from the upper- or lower-triangles, and as such there will be duplicate, flipped similarities in the result set).</small>

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
The library also has supoprt for term frequency and inverse document frequency functionality.

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
| count          | ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/69faba5875c1ba7d6a3820c813ba22fba35185f5)        | idf           | ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/864fcfdc0c16344c11509f724f1aa7081cf9f657)         |
| binary         | ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/a019735e07635e5a74673d6e1a34919027e645f5)        | smooth        | ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/25f4d6690acaaef1f15f308d24f6f8a439de971d)         |
| freq           | ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/91699003abf4fe8bdf861bbce08e73e71acf5fd4)        | max           | ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/f15c125a1d7f1327afeecc4e2b89272a9a094338)         |
| lognorm        | ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/9c173382612c58c00325c4e9f593739ab3afc324)        | probabilistic | ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/1868194cba8431aa2d556dd1aac90d78833eaaf3)         |
| doublenormhalf | ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/45badc1c70ec2caa00ed8c21ed75bd9f8d3e650c)        |               |          |
| doublenormk    | ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/65b776d7a3f8e42f15c880fb7582282b987684fe)        |               |          |


Users may specify a specific function for the term frequency or inverse document frequency as well. The function will be executed on the term-document matrix using the numpy functions `apply_along_axis(func, 1, dtm)` and `apply_along_axis(func, 0, dtm)` respectively.

```python
# example of weighting scheme 2 from the wikipedia article
tf = lambda x: 1+np.log(x)
idf = lambda x: np.log(1+(dtm.DTM.shape[0]/np.count_nonzero(x[x>0])))

dtm = dtm.DocumentTermMatrix(tf_func=tf, idf_func=idf)
```