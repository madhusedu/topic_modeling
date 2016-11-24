from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
#from gensim import corpora, models
#import gensim
import pandas
import random
from itertools import chain
import numpy
#from nltk.stem.snowball import SnowballStemmer

k1 = 2 # number of topics
alpha = 1 # hyperparameter. single value indicates symmetric dirichlet prior. higher=>scatters document clusters
eta = 0.001 # hyperparameter
iterations = 3 # iterations for collapsed gibbs sampling.  This should be a lot higher than 3 in practice.

#taking the data from csv files and naming the columns
column_names = ['a', 'b', 'c', 'd', 'e','f','g','h','i','j']

#reading the data using the pandas library
fileread = pandas.read_csv('review_data.csv', names=column_names)


reviews = fileread.c.tolist()
review =  [i.decode('UTF-8') if isinstance(i, basestring) else i for i in reviews]
tokenizer = RegexpTokenizer(r'\w+')

#we are using the stop_words to implement this method
int_stop = get_stop_words('en')

#we are using porterstemmer to implement this method
p_stemmer = PorterStemmer()
#stemmer2 = SnowballStemmer("english", ignore_stopwords=True)

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
print ""
for j in range(0,len(texts)):
    for k in range(0,len(texts[j])):
        while(texts[j][k]in docs):
            texts[j][k]= docs.index(texts[j][k])

print "word ID"
print ""
print texts
print ""


#wt=matrix = [[0 for i in range(K)] for j in range(len(docs))]
wt=numpy.zeros((k1,len(docs)))
#wt=numpy.zeros((3,20))
print wt

ta=[]
for i in range(len(texts)):
    ta.append([])
    for j in range(len(texts[i])):
        ta[i].append(0)

print ta

for d in range(len(texts)):
    for w in range(len(texts[d])):
        ta[d][w]=random.randint(1, k1)
        ti = ta[d][w] # topic index - check ta[[d]] is changed to ta[d][w]
        wi = texts[d][w] # wordID for token w
        #wt[ti][wi] = wt[ti][wi]
print ta
