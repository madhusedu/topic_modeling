rawdocs = ('eat turkey on turkey day holiday',
            'i like to eat cake on holiday',
            'turkey trot race on thanksgiving holiday',
            'snail race the turtle',
            'time travel space race',
            'movie on thanksgiving',
            'movie at air and space museum is cool movie',
            'aspiring movie star')

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

#print docs
print wordfreq
