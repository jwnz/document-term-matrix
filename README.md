### DTM

This is an incomplete document-term matrix python <i>library</i>.



Initialize the model, load the text data (list of lists of strings), and run the build function.
```python
from document_term_matrix import DocumentTermMatrix, utils

dtm = DocumentTermMatrix.DocumentTermMatrix()
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



Calculate the similarity between any two words in the vocab:
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

Some more convoluted examples using the 2D numpy array include mapping a word in the vocab to it's corresponding vector:

```python
{dtm.vocab[i]:vec for i,vec in enumerate([dtm.DTM[:, i] for i in range(len(dtm.vocab))])}

>>> {'뱅코우': array([0., 0., 0., ..., 0., 0., 0.]),
     '뱅크': array([0., 0., 0., ..., 0., 0., 0.]),
     '뱅킹': array([0., 0., 0., ..., 0., 0., 0.]),
     '버': array([0., 0., 0., ..., 0., 0., 0.]),
     '버거': array([0., 0., 0., ..., 0., 0., 0.]),
     '버거킹': array([0., 0., 0., ..., 0., 0., 0.]),
     '버건디': array([0., 0., 0., ..., 0., 0., 0.]),
     '버그': array([0., 0., 0., ..., 0., 0., 0.]),
     '버그달': array([0., 0., 0., ..., 0., 0., 0.]),
     '버그만': array([0., 0., 0., ..., 0., 0., 0.])}
```

or calculating the augmented frequency tf-idf for each word:
```python
import numpy as np

# working with the 4th document
n = 4

# calculate term frequency
tf = dtm.DTM[n, :]/dtm.DTM[n, :].sum()

# calculate the inverse document frequency
idf = np.log(dtm.DTM.shape[0]/np.sum(dtm.DTM>0, axis=0))

# multiply tf by idf to calculate the tf-idf
tfidf = tf*idf

tfidf
>>> array([5.1547438 , 5.1547438 , 5.1547438 , ..., 4.18347412, 4.35088595, 5.15554622])
```

And perhaps display everything in a neat Pandas DataFrame:
```python
import pandas as pd
df = pd.DataFrame([{'docnum':n, 'word':dtm.vocab[i],'tfidf':val} for i,val in enumerate(tfidf)])
>>> df

   docnum   word     tfidf
0    4      뱅코우   5.154744
1    4       뱅크    3.366917
2    4       뱅킹    4.058600
3    4        버     3.914662
4    4       버거    3.639000
5    4      버거킹   5.155011
6    4      버건디   4.462255
7    4       버그    3.428355
8    4      버그달   5.154744
9    4      버그만   5.154744
```