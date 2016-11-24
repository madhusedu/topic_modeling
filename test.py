rawdocs = ('eat turkey on turkey day holiday',
            'i like to eat cake on holiday',
            'turkey trot race on thanksgiving holiday',
            'snail race the turtle',
            'time travel space race',
            'movie on thanksgiving',
            'movie at air and space museum is cool movie',
            'aspiring movie star')

word_id = None

def assign_id(warray):
    global word_id
    wordlist=[]
    count = 0
    wordlist.append(warray[0])
    for i in range(0,len(warray)):
        flag = 0
        for j in range(0, len(wordlist)):
            if warray[i] == wordlist[j]:
                flag+=1
        if flag == 0:
            wordlist.append(warray[i])

    temp = ((i, wordlist[i]) for i in range(0, len(wordlist)))

    word_id=dict(temp)
    word_id={v: k for k, v in word_id.iteritems()}


    for i in range(0, len(warray)):
       warray[i] = word_id[str(warray[i])]

    return warray


docs = []
for i in range(0, len(rawdocs)):
    attach = rawdocs[i].split()
    for j in range(0, len(attach)):
        docs.append(attach[j])

assign_id(docs)

wordfreq = []
for w in docs:
    wordfreq.append(docs.count(w))


#print docs
print
print wordfreq

word_id={v: k for k, v in word_id.iteritems()}

for i in range(0,len(word_id)):
    print str(word_id[i]) + '   ' + str(wordfreq[i])