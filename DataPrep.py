
import codecs
import pickle

with codecs.open('dataset/original/spam.txt', "r",encoding='utf-8', errors='ignore') as file:
    raw_spam = file.readlines()

vocab = set()
    
spam = list()
for row in raw_spam:
    spam.append(set(row[:-2].split(" ")))
    for word in spam[-1]:
        vocab.add(word)
    
import codecs
with codecs.open('dataset/original/ham.txt', "r",encoding='utf-8', errors='ignore') as file:
    raw_ham = file.readlines()

ham = list()
for row in raw_ham:
    ham.append(set(row[:-2].split(" ")))
    for word in ham[-1]:
        vocab.add(word)
        
vocab.add("<unk>")

vocab = list(vocab)
with open("dataset/vocab.txt", "wb") as f:   
   pickle.dump(vocab, f)

w2i = {}
for i,w in enumerate(vocab):
    w2i[w] = i
    
def to_indices(input, l=500):
    indices = list()
    for line in input:
        if(len(line) < l):
            line = list(line) + ["<unk>"] * (l - len(line))
            idxs = list()
            for word in line:
                idxs.append(w2i[word])
            indices.append(idxs)
    return indices
            
spam_idx = to_indices(spam)
ham_idx = to_indices(ham)

train_spam_idx = spam_idx[0:-1000]
train_ham_idx = ham_idx[0:-1000]

test_spam_idx = spam_idx[-1000:]
test_ham_idx = ham_idx[-1000:]

train_data = list()
train_target = list()

test_data = list()
test_target = list()

for i in range(max(len(train_spam_idx),len(train_ham_idx))):
    train_data.append(train_spam_idx[i%len(train_spam_idx)])
    train_target.append([1])
    
    train_data.append(train_ham_idx[i%len(train_ham_idx)])
    train_target.append([0])
    
for i in range(max(len(test_spam_idx),len(test_ham_idx))):
    test_data.append(test_spam_idx[i%len(test_spam_idx)])
    test_target.append([1])
    
    test_data.append(test_ham_idx[i%len(test_ham_idx)])
    test_target.append([0])

with open("dataset/train/train_data.txt", "wb") as f:   
   pickle.dump(train_data, f)
with open("dataset/train/train_target.txt", "wb") as f:   
   pickle.dump(train_target, f)
with open("dataset/test/test_data.txt", "wb") as f:   
   pickle.dump(test_data, f)
with open("dataset/test/test_target.txt", "wb") as f:   
   pickle.dump(test_target, f)

with open("dataset/Client_1/train_data.txt", "wb") as f:   
   pickle.dump(train_data[0:1000], f)
with open("dataset/Client_1/train_target.txt", "wb") as f:   
   pickle.dump(train_target[0:1000], f)
with open("dataset/Client_2/train_data.txt", "wb") as f:   
   pickle.dump(train_data[1000:2000], f)
with open("dataset/Client_2/train_target.txt", "wb") as f:   
   pickle.dump(train_target[1000:2000], f)
with open("dataset/Client_3/train_data.txt", "wb") as f:   
   pickle.dump(train_data[2000:], f)
with open("dataset/Client_3/train_target.txt", "wb") as f:   
   pickle.dump(train_target[2000:], f)
