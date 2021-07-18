
from Layer import Embedding
import pickle
from Train_and_Test import train, test
import copy


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


for i in range(3):
    print("Starting Training Round...")
    print("\tStep 1: send the model to Client_1")
    Client_1_model = train(copy.deepcopy(model), Client_1_data, Client_1_target, iterations=1)
    
    print("\n\tStep 2: send the model to Client_2")
    Client_2_model = train(copy.deepcopy(model), Client_2_data, Client_2_target, iterations=1)
    
    print("\n\tStep 3: Send the model to Client_3")
    Client_3_model = train(copy.deepcopy(model), Client_3_data, Client_3_target, iterations=1)
    
    print("\n\tAverage Everyone's New Models")
    model.weight.data = (Client_1_model.weight.data + \
                         Client_2_model.weight.data + \
                         Client_3_model.weight.data)/3
    
    print("\t% Correct on Test Set: " + \
          str(test(model, test_data, test_target)*100))
    
    print("\nRepeat!!\n")
