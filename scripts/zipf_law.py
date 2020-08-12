import nltk
from collections import Counter
nltk.data.path.append('/home/leo/Documents/machineLearningDatasets/nltk_data')

from nltk.corpus import brown
brown.words()[:5]
brown.tagged_words()[:5]
puncs = set((',', '.', '--', '-', '!', '?', ':', ';', '(', ')',
             '[', ']', '``', "''"))
word_list = (x.lower() for x in brown.words() if x not in puncs)
token_counts = Counter(word_list)
token_counts.most_common(5)
