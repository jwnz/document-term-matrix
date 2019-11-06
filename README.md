### DTM

This is an incomplete document-term matrix python <i>library</i>.



Initialize the model, load the text data (list of lists of strings), and run the build function.
```
from DTM import model, utils

dtm = model.DTM()
sentences = utils.DocumentIterator('preprocessed_text_data.txt', delim='\t')
dtm.build(sentences)
```

Check the word frequencies as follows:
```
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
```
dtm.word_2_word_sim('the', 'with')
>>> 0.18393539930448133
```

Compute the parwise (cosine) similarity for all word pairs in the vocab.<br>
This generally requires a lot of memory, so it's recommended to set the <i>cutoff</i> parameter to filter out words with
low frequencies to reduce the size of the matrix. Additionally, the <i>tolerance</i> parameter can be used to filter out any words pairs with a similarity below the given value:
```
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

The Document-Term Matrix is a numpy 2D array.

