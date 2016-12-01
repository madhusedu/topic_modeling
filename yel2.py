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

def dt_alpha_mat(dt_alpha_array):
        dt_alpha_rowsum=[]
        for i in range(len(dt_alpha_array)):
            dt_alpha_rowsum1= 0
            for j in range(len(dt_alpha_array[i])):
                dt_alpha_rowsum1=dt_alpha_rowsum1+dt_alpha_array[i][j]
            dt_alpha_rowsum.append(dt_alpha_rowsum1)
        return dt_alpha_rowsum

def wt_eta_mat(wt_eta_array):
    wt_eta_rowsum=[]
    for i in range(len(wt_eta_array)):
        wt_eta_rowsum1=0
        for j in range(len(wt_eta_array[i])):
            wt_eta_rowsum1=wt_eta_rowsum1+wt_eta_array[i][j]
        wt_eta_rowsum.append(wt_eta_rowsum1)
    return wt_eta_rowsum

def function(phi1,n):
    term=numpy.zeros((n,k1),dtype=numpy.float)
    sorted_phi1=numpy.array(phi1)
    print "phi1"
    print phi1
    for p in range(len(phi1)):
        phi1.sort_index()
        #phi1.reverse()
    print "phi1.sort()"
    print phi1
    print "sorted_phi1[p]"
    print sorted_phi1[p]
    print "term"
    print term
    return 1




k1 = 3 # number of topics
alpha = 1 # hyperparameter. single value indicates symmetric dirichlet prior. higher=>scatters document clusters
eta = 0.001 # hyperparameter
iterations = 3 # iterations for collapsed gibbs sampling.  This should be a lot higher than 3 in practice.
n=4 #Top n words belonging to a topic

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

newlist=[[]]
df = pandas.read_csv('/Users/bharathnagesh/Documents/topic_modeling/AFINN.csv', names=['word', 'score'])
arr = numpy.array(df)
wordAFINN = list(arr[:,0])
#print x
weightAFINN = list(arr[:,1])
#print x_
#x1 =  [i.decode('UTF-8','ignore') if isinstance(i, basestring) else i for i in x]
#print x1
#final = [i for i in docs if i in x]
p,q = 0,0
temp = [[] for i in range(len(texts))]
for i in range(0,len(texts)):
    for j in range(0, len(texts[i])):
        for k in range(0, len(wordAFINN)):
            if texts[i][j] == wordAFINN[k]:
                temp[p].append(wordAFINN[k])

    p = p + 1
texts = temp


#for j in range(0,len(texts)):
#    for k in range(0,len(texts[j])):
#        while(texts[j][k]in x1):
#            newlist[j].append(texts[j][k])


print "newlist"
print newlist
print "preprocessed text-TEXTS"
print ""
print texts
print ""

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
            p_z_sum=p_z.sum()

            t1_array=numpy.random.choice(k1, size=1, replace=True, p=p_z/p_z_sum)
            t1=t1_array[0]
            ta[d][w] = t1+1
            dt[d][t1]=dt[d][t1]+1
            wt[t1][wid] = wt[t1][wid]+1


dt_array=numpy.array(dt)
dt_alpha_array=dt_array+alpha
#wt_wid_value=wt_mat(wid)
dt_alpha_rowsum=dt_alpha_mat(dt_alpha_array)
dt_alpha_rowsum_vector=numpy.array(dt_alpha_rowsum, dtype=float)
theta=dt_alpha_array/dt_alpha_rowsum_vector[:,None]

wt_array=numpy.array(wt)
wt_eta_array=wt_array+eta

wt_eta_rowsum=wt_eta_mat(wt_eta_array)
wt_eta_rowsum_vector=numpy.array(wt_eta_rowsum, dtype=float)
phi=wt_eta_array/wt_eta_rowsum_vector[:,None]

for i in range(0, len(phi)):
    for j in range(0, len(phi[i])):
            for k in range(0, len(wordAFINN)):
                if docs[j] == wordAFINN[k]:
                    phi[i][j] = phi[i][j] * weightAFINN[k]

sums = []
temp_sum=0

for i in range(0, len(phi)):
    for j in range(0, len(phi[i])):
        temp_sum = temp_sum + phi[i][j]
    sums.append(temp_sum)
    temp_sum = 0

print "SUMS"
print sums
print ""

final_score = [] #because number of reviews = 6
temp_sum = 0
pos = 0
max_num = -numpy.inf

for i in range(0, len(theta)):
    for j in range(0, len(theta[i])):
        if max_num < theta[i][j]:
            max_num = theta[i][j]
            pos = j
    temp_sum = max_num* sums[pos]
    final_score.append(temp_sum)
    max_num = -numpy.inf

print "Final Score"
print final_score
print ""


df=pandas.DataFrame(phi)
df.columns=docs

print "df"
print df

phi1=df





print "phi1"
print phi1
print "phi"
print phi

print "theta"
print theta
print "dt_alpha_array"
print dt_alpha_array
print "dt_alpha_rowsum"
print dt_alpha_rowsum
print "dt_alpha_rowsum_vector"
print dt_alpha_rowsum_vector






print "dt"
print dt
print "dt_array+alpha"
print dt_array+alpha
print "dt_alpha_array"
print dt_alpha_array
print t1
print "p_z"
print p_z
print "p_z_sum"
print p_z_sum
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

theta_array=numpy.array(theta)
topic_inter=[]
for i in range(len(theta_array)):
    max_index=numpy.argmax(theta_array[i])
    topic_inter.append(max_index)
topic=numpy.array(topic_inter)
topic=topic+1

print "topic"
print topic

terms=function(phi1,n)
