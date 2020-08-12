from nlpia.data.loaders import harry_docs as docs
from nltk.tokenize import TreebankWordTokenizer
from collections import OrderedDict

doc_tokens = []
tokenizer = TreebankWordTokenizer()
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
all_doc_tokens = sum(doc_tokens, [])
lexicon = sorted(set(all_doc_tokens))
zero_vector = OrderedDict((token, 0) for token in lexicon)
