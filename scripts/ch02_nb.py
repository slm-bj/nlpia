# Codes in section 2.3.2: Naive Bayes

from nlpia.data.loaders import get_data
import pandas as pd
from nltk.tokenize import casual_tokenize
from collections import Counter
from sklearn.naive_bayes import MultinomialNB

movies = get_data('hutto_movies')
movies.head().round(2)
movies.describe()
pd.set_option('display.width', 75)
bags_of_words = []
for text in movies.text:
    bags_of_words.append(Counter(casual_tokenize(text)))
df_bows = pd.DataFrame.from_records(bags_of_words)
df_bows = df_bows.fillna(0).astype(int)

nb = MultinomialNB()
nb = nb.fit(df_bows, movies.sentiment > 0)
movies['predicted_sentiment'] = nb.predict_proba(df_bows)[:, 1] * 8 - 4
movies['error'] = (movies.predicted_sentiment - movies.sentiment).abs()

movies['sentiment_ispositive'] = (movies.sentiment > 0).astype(int)
movies['predicted_ispositive'] = (movies.predicted_sentiment > 0).astype(int)
movies['sentiment predicted_sentiment sentiment_ispositive '
       'predicted_ispositive'.split()].head(8)
(movies.predicted_ispositive ==
 movies.sentiment_ispositive).sum() / len(movies)


products = get_data('hutto_products')
bows = []
for text in products.text:
    bows.append(Counter(casual_tokenize(text)))

df_pbows = pd.DataFrame.from_records(bows)
df_pbows = df_pbows.fillna(0).astype(int)
# df_all_bows = df_bows.append(df_pbows)

nbp = MultinomialNB().fit(df_pbows, products.sentiment > 0)
products['predicted_sentiment'] = nbp.predict_proba(df_pbows)[:, 1] * 8 - 4
products['sentiment_ispositive'] = (products.sentiment > 0).astype(int)
products['predicted_ispositive'] = (products.predicted_sentiment
                                    > 0).astype(int)
(products.predicted_ispositive ==
 products.sentiment_ispositive).sum() / len(products)
