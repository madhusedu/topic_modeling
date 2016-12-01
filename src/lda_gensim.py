from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from itertools import chain
import gensim
import pandas

#taking the data from csv files and naming the columns
column_names = ['a', 'b', 'c', 'd', 'e','f','g','h','i','j']

#reading the data using the pandas library
fileread = pandas.read_csv('/data/test_data.csv', names=column_names)


reviews = fileread.c.tolist()
review =  [i.decode('UTF-8') if isinstance(i, basestring) else i for i in reviews]
tokenizer = RegexpTokenizer(r'\w+')

#we are using the stop_words to implement this method
int_stop = get_stop_words('en')

#we are using porterstemmer to implement this method
p_stemmer = PorterStemmer()



texts = []


for i in review:

    #converting to lower case
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in int_stop]
    #print "Stopped tokens"
    #print stopped_tokens

    # stem tokens
    #stemmed_tokens = [stemmer2.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stopped_tokens)

    #print "preprocessed text"
    #print texts

print "preprocessed text-TEXTS"
print ""
print texts
print ""
indexvalue=1;

# Replace words in documents with wordIDs
docs=list(chain.from_iterable(texts))
print "flattened matrix- DOCS"
print ""
print docs
print ""# turn our tokenized documents into a id <-> term dictionary


dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

print "Corpus "
print corpus

# split into 80% training and 20% test sets
p = int(len(corpus) * .5)
cp_train = corpus[0:p]
cp_test = corpus[p:]
# generate LDA model
model = gensim.models.ldamodel.LdaModel(corpus=cp_train, num_topics=5, id2word = dictionary, passes=50)

perplex = model.bound(cp_test)
print "Perplexity: %s" % perplex

print "corect--------------------------------------------------"
print "\n"
print model
print "LDA model"
print(model.print_topics(num_topics=7, num_words=5))
