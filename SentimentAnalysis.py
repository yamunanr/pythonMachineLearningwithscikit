import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'This is the first document.',
    'This is the second document.',
    'This is the third document. Document number three',
    'Number four. To repeat, number four']

vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(corpus)

bag_of_words

print(bag_of_words)

 #Access ID corresponding to word by:

acc_ID = vectorizer.vocabulary_.get('repeat')
print(acc_ID)

print(vectorizer.vocabulary_)


#tf-idf vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
bag_of_words = vectorizer.fit_transform(corpus)
print(bag_of_words)

#acess ID for words using vectorizer or using dataframe pandas
print(vectorizer.vocabulary_.get('number'))
words_as = pd.DataFrame(bag_of_words.toarray(), columns=vectorizer.get_feature_names())
print(words_as)


# incase of large vocabulary of corpus us HashingVectorizer

from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(n_features=8)   # value 8 is set for 8 size of
#leading to word IDs are from 0 to 7 - multipl words may hash to the same value
#frequesncy represented is normalized form
feature_vector = vectorizer.fit_transform(corpus)
print(feature_vector)
