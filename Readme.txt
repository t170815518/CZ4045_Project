*This project is for Assignment 1, Course CZ4045: NLP at NTU*
# Dependency
```
numpy~=1.21.3
gsdmm~=0.1
setuptools~=58.2.0
nltk~=3.6.5
spacy~=3.1.3
seaborn~=0.11.2
matplotlib~=3.4.3
scikit-learn~=1.0
tqdm~=4.62.3
wordcloud~=1.8.1
```

To install the dependencies, run
```
pip install -r requirements.txt
```
# Usage
For Jupyter Notebook files, start Jupyter Notebook server and run all codes.
+ Most frequent Noun - Adjective pairs for each rating.ipynb
+ indicativeADJP.ipynb
+ Tokenization and Stemming.ipynb

For writing style analysis, run `python writing_style.py`. The program outputs the analysis of different domains separately, e.g.
the error identified by the grammar checker
```
=====================Analyze ['data/hardware_zone_1.txt', 'data/hardware_zone_2.txt'] ===============================
Offset 46, length 4, Rule ID: COMMA_COMPOUND_SENTENCE
Message: Use a comma before ‘and’ if it connects two independent clauses (unless they are closely connected and short).
Suggestion: , and
...flagship-class phone has an OLED display and it looks like thatâ€™s going to be true...
```

For POS tagging, run `python POS_tagging.py`. The program outputs the result
produced by two tokenizers, i.e., unigram and perceptron:
```
The sentence chosen is:
 Response I got was "because it's ethnic hair, we had to charge the $55".
----------------------------------------
POS tagging using unigram:
 [('Response', 'NN'), ('I', 'PPSS'), ('got', 'VBD'), ('was', 'BEDZ'), ('``', '``'), ('because', 'CS'), ('it', 'PPS'), ("'s", None), ('ethnic', 'JJ'), ('hair', 'NN'), (',', ','), ('we', 'PPSS'), ('had', 'HVD'), ('to', 'TO'), ('charge', 'NN'), ('the', 'AT'), ('$', None), ('55', 'CD'), ("''", "''"), ('.', '.')]
POS tagging using nltk:
 [('Response', 'NN'), ('I', 'PRP'), ('got', 'VBD'), ('was', 'VBD'), ('``', '``'), ('because', 'IN'), ('it', 'PRP'), ("'s", 'VBZ'), ('ethnic', 'JJ'), ('hair', 'NN'), (',', ','), ('we', 'PRP'), ('had', 'VBD'), ('to', 'TO'), ('charge', 'VB'), ('the', 'DT'), ('$', '$'), ('55', 'CD'), ("''", "''"), ('.', '.')]

```


## Application Usage
To run topic modeling of a business, an example is
```
python review_analyzer.py --business AktuBx1W7c3ZdzwuaOp8xg --topic_num 10 --is_visualize True
```
+ `--business`: str, the index of business to analyze. Empty string by default, which means
the complete dataset is analyzed as a whole.
+ `--topic_num`: a positive integer, the number of topics to modeled.
+ `--is_visualize`: bool, True means to visualize the word clouds and export them locally
### Output
The example of console output:
```
Started Reading JSON file which contains multiple JSON objects
Load dataset: 100%|██████████| 15300/15300 [00:04<00:00, 3362.62it/s]
Modelling topic with LDA...
Topic 0:  ['also', 'place', 'even', 'well', 'chinese', 'little', 'chicken', 'order', 'like', 'food']
Topic 1:  ['soup', 'lunch', 'order', 'place', 'great', 'little', 'chicken', 'good', 'chinese', 'food']
Topic 2:  ['walnut', 'deliver', 'friendly', 'rice', 'fried', 'chinese', 'pork', 'restaurants', 'dumpling', 'shrimp']
Topic 3:  ['wont', 'back', 'quick', 'large', 'recommended', 'affordable', 'highly', 'ok', 'flavorless', 'soups']
Topic 4:  ['back', 'meat', 'crab', 'get', 'like', 'ive', 'ordered', 'food', 'beef', 'place']
Topic 5:  ['like', 'fried', 'spicy', 'chinese', 'order', 'thai', 'good', 'food', 'chicken', 'place']
Topic 6:  ['salt', 'driver', 'delicious', 'eat', 'service', 'place', 'really', 'shrimp', 'restaurant', 'food']
Topic 7:  ['dumplings', 'better', 'considering', 'least', 'long', 'soup', 'wait', 'order', 'food', 'chicken']
Topic 8:  ['get', 'sour', 'go', 'sauce', 'little', 'chicken', 'delivery', 'order', 'good', 'food']
Topic 9:  ['restaurants', 'cup', 'eat', 'food', 'thing', 'cheese', 'items', 'sauce', 'msg', 'parmesan']
=======================Modelling topic with LDA Completes=================
Modelling topic with GSDMM...
In stage 0: transferred 90 clusters with 21 clusters populated
In stage 1: transferred 47 clusters with 13 clusters populated
In stage 2: transferred 25 clusters with 9 clusters populated
In stage 3: transferred 17 clusters with 8 clusters populated
In stage 4: transferred 15 clusters with 8 clusters populated
In stage 5: transferred 9 clusters with 9 clusters populated
In stage 6: transferred 14 clusters with 9 clusters populated
In stage 7: transferred 11 clusters with 9 clusters populated
In stage 8: transferred 10 clusters with 9 clusters populated
In stage 9: transferred 13 clusters with 8 clusters populated
Topic 1:  []
Topic 12:  []
Topic 24:  ['thai', 'pad', 'way', 'eat', 'gf', 'seeing', 'ordered', 'place', 'really', 'wanted']
Topic 25:  ['excellent', 'items', 'tried', 'fantastic', 'thi', 'salad', 'shrimp', 'yummy', 'chicken', 'beef']
Topic 14:  ['tea', 'plastic', 'served', 'cheap', 'cup', 'even', 'place', 'food', 'used', 'plates']
Topic 11:  ['kung', 'better', 'food', 'sauce', 'order', 'good', 'rice', 'fried', 'lemon', 'chicken']
Topic 10:  ['today', 'first', 'service', 'restaurant', 'order', 'went', 'back', 'time', 'food', 'driver']
Topic 3:  ['like', 'eat', 'soup', 'even', 'ever', 'tasted', 'little', 'im', 'chicken', 'food']
Topic 0:  ['service', 'delicious', 'always', 'ordered', 'good', 'friendly', 'ive', 'chinese', 'place', 'food']
Topic 29:  ['go', 'like', 'shrimp', 'little', 'place', 'chicken', 'chinese', 'order', 'good', 'food']
```

The word clouds are exported as `png`file in the current directory, e.g.
`LDA_8.png` and `GSDMM_16.png`