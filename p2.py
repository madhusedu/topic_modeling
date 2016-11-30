from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import pandas
import random
from itertools import chain
import numpy

def denom_a_func(d):
    denom_a_sum1= 0
    for j in range(len(dt[d])):
        denom_a_sum1=denom_a_sum1+dt[d][j]
    return denom_a_sum1

def denom_b_func(dt):
    denom_b_sum=[]
    for i in range(len(dt)):
        denom_b_sum1= 0
        for j in range(len(dt[i])):
            denom_b_sum1=denom_b_sum1+dt[i][j]
        denom_b_sum.append(denom_b_sum1)
    return denom_b_sum

def wt_mat(wid):
    wt_wid_value=[]
    for i in range(len(wt)):
        wt_wid_value.append(wt[i][wid])
    return wt_wid_value

def dt_mat(d):
    dt_d_value=[]
    for i in range(len(dt[d])):
        dt_d_value.append(dt[d][(i)])
    return dt_d_value


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

#stop_words used for implementation
int_stop = get_stop_words('en')

#porterstemmer used for implementation
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

print "word ID-texts"
print ""
print texts
print ""


#wt=matrix = [[0 for i in range(K)] for j in range(len(docs))]
wt=numpy.zeros((k1,len(docs)),dtype=numpy.int)
#wt=numpy.zeros((3,20))
print
print "wt"
print wt

ta=[]
for i in range(len(texts)):
    ta.append([])
    for j in range(len(texts[i])):
        ta[i].append(0)
print
print "ta"
print ta

for d in range(len(texts)):
    for w in range(len(texts[d])):
        ta[d][w]=random.randint(1, k1)
        ti = ta[d][w] # topic index - check ta[[d]] is changed to ta[d][w]
        wi = texts[d][w] # wordID for token w
        wt[ti-1][wi] = wt[ti-1][wi]+1
print "ta"
print ta
print
print "wt"
print wt

print
print "ta"
print ta



dt=numpy.zeros((len(texts),k1),dtype=numpy.int)
print dt
#for(d in 1:length(docs)){ # for each document d
for d in range(len(texts)):
  #for(t in 1:K){ # for each topic t
  for t in range(1,k1+1):
      for col in range(len(texts[d])):
        if(ta[d][col]==t):
            dt[d][t-1]=dt[d][t-1]+1;
    #dt[d][t]=sum(1 if(ta[d]==t))
    #dt[d,t] <- sum(ta[d]==t) # count tokens in document d assigned to topic t

print

print "dt"
print dt

denom_b=[] #define denom_b cause using it inside the module
wt_wid_value_eta=[] #define wt_wid_value_eta cause using it inside the module
dt_d_value_alpha=[] #define dt_d_value_alpha cause using it inside the module
for i in range(0,iterations):
    for d in range(0,len(texts)):
        for w in range(0,len(texts[d])):
            t0=ta[d][w]
            wid=texts[d][w]

            dt[d][t0-1]=dt[d][t0-1]-1
            wt[t0-1][wid] = wt[t0-1][wid]-1

            #denom_a = numpy.sum(dt, axis=1) + (K1) * alpha # number of tokens in document + number topics * alpha
            denom_a_sum=denom_a_func(d)
            denom_a=denom_a_sum+k1*alpha

            denom_b_sum=denom_b_func(wt)

            denom_b[:]=[x+(len(docs) * eta) for x in denom_b_sum]

            wt_wid_value=wt_mat(wid)
            wt_wid_value_eta[:]=[x+(eta) for x in wt_wid_value]

            dt_d_value=dt_mat(d)
            dt_d_value_alpha[:]=[x+(alpha) for x in dt_d_value]

            #p_z=wt_wid_value_eta/denom_b*dt_d_value_alpha/denom_a
            r1=numpy.array(wt_wid_value_eta)
            r2=numpy.array(denom_b)
            r3=numpy.array(dt_d_value_alpha)
            r4=numpy.array(denom_a)

            p_z=r1/r2*r3/r4








print "p_z"
print p_z
print "wt_wid_value"
print wt_wid_value
print "wt_wid_value_eta"
print wt_wid_value_eta
print "dt_d_value"
print dt_d_value
print "dt_d_value_alpha"
print dt_d_value_alpha
print "denom_a"
print denom_a
print "denom_b"
print denom_b_sum
print denom_b
            #denom_a = numpy.sum(dt[d], axis=1)
            #print "loop denom_a"
            #print denom_a

            #denom_b = rowSums(wt) + len(docs) * eta # number of tokens in each topic + # of words in vocab * eta
