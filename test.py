rawdocs = ('eat turkey on turkey day holiday',
            'i like to eat cake on holiday',
            'turkey trot race on thanksgiving holiday',
            'snail race the turtle',
            'time travel space race',
            'movie on thanksgiving',
            'movie at air and space museum is cool movie',
            'aspiring movie star')

def assign_id(warray):
    word_id =[]

    for i in range(0,len(warray)):
        flag = 0
        for j in range(0, len(word_id)):
            if warray[i] == word_id[j]:
                flag+=1
        if flag == 0:
            word_id.append(warray[i])

    print word_id

docs = []
for i in range(0, len(rawdocs)):
    attach = rawdocs[i].split()
    for j in range(0, len(attach)):
        docs.append(attach[j])

wordfreq = []
for w in docs:
    wordfreq.append(docs.count(w))

K = 2
alpha = 1
eta = .001
iterations = 3

assign_id(docs)

#print wordfreq
