
import numpy as np
import pickle
from Train_and_Test import train
from Layer import Embedding
from DataPrep import w2i
import copy

with open("dataset/vocab.txt", "rb") as f:   
   vocab = pickle.load(f)


Client_email = ["client", "data", "is" ,"at", "risk"]

Client_input = np.array([[w2i[x] for x in Client_email]])
Client_target = np.array([[0]])

sent_model = Embedding(vocab_size=len(vocab), dim=1)
sent_model.weight.data *= 0


Client_model = train(copy.deepcopy(sent_model), Client_input, Client_target, batch_size=1,iterations=1)

for i, v in enumerate(Client_model.weight.data - sent_model.weight.data):
    if(v != 0):
        print(vocab[i])