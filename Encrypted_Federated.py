import phe
from Layer import Embedding
import pickle
from Train_and_Test import train, test
import copy
import numpy as np


with open("dataset/vocab.txt", "rb") as f:   
   vocab = pickle.load(f)
with open("dataset/Client_1/train_data.txt", "rb") as f:   
   Client_1_data = pickle.load(f)
with open("dataset/Client_1/train_target.txt", "rb") as f:   
   Client_1_target = pickle.load(f)
with open("dataset/Client_2/train_data.txt", "rb") as f:   
   Client_2_data = pickle.load(f)
with open("dataset/Client_2/train_target.txt", "rb") as f:   
   Client_2_target = pickle.load(f)
with open("dataset/Client_3/train_data.txt", "rb") as f:   
   Client_3_data = pickle.load(f)
with open("dataset/Client_3/train_target.txt", "rb") as f:   
   Client_3_target = pickle.load(f)
with open("dataset/test/test_data.txt", "rb") as f:   
   test_data = pickle.load(f)
with open("dataset/test/test_target.txt", "rb") as f:   
   test_target = pickle.load(f)
           
model = Embedding(vocab_size=len(vocab), dim=1)
model.weight.data *= 0

# note that in production the n_length should be at least 1024
public_key, private_key = phe.generate_paillier_keypair(n_length=128)

def train_and_encrypt(model, input, target, pubkey):
    new_model = train(copy.deepcopy(model), input, target, iterations=1)

    encrypted_weights = list()
    for val in new_model.weight.data[:,0]:
        encrypted_weights.append(public_key.encrypt(val))
    ew = np.array(encrypted_weights).reshape(new_model.weight.data.shape)
    
    return ew

for i in range(3):
    print("\nStarting Training Round...")
    print("\tStep 1: send the model to Client_1")
    Client_1_encrypted_model = train_and_encrypt(copy.deepcopy(model),Client_1_data, Client_1_target, public_key)

    print("\n\tStep 2: send the model to Client_2")
    Client_2_encrypted_model = train_and_encrypt(copy.deepcopy(model),Client_2_data, Client_2_target, public_key)

    print("\n\tStep 3: Send the model to Client_3")
    Client_3_encrypted_model = train_and_encrypt(copy.deepcopy(model),Client_3_data, Client_3_target, public_key)

    print("\n\tStep 4: All 3 Clients send their")
    print("\tencrypted models to each other.")
    aggregated_model = Client_1_encrypted_model + \
                       Client_2_encrypted_model + \
                       Client_3_encrypted_model

    print("\n\tStep 5: only the aggregated model")
    print("\tis sent back to the model owner who")
    print("\t can decrypt it.")
    raw_values = list()
    for val in Client_3_encrypted_model.flatten():
        raw_values.append(private_key.decrypt(val))
    model.weight.data = np.array(raw_values).reshape(model.weight.data.shape)/3

    print("\t% Correct on Test Set: " + \
              str(test(model, test_data, test_target)*100))