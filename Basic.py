
from Layer import Embedding
from Loss import MSELoss
from Gradient import SGD
import pickle
from Train_and_Test import train, test


with open("dataset/vocab.txt", "rb") as f:   
   vocab = pickle.load(f)
with open("dataset/train/train_data.txt", "rb") as f:   
   train_data = pickle.load(f)
with open("dataset/train/train_target.txt", "rb") as f:   
   train_target = pickle.load(f)
with open("dataset/test/test_data.txt", "rb") as f:   
   test_data = pickle.load(f)
with open("dataset/test/test_target.txt", "rb") as f:   
   test_target = pickle.load(f)

model = Embedding(vocab_size=len(vocab), dim=1)
model.weight.data *= 0
criterion = MSELoss()
optim = SGD(parameters=model.get_parameters(), alpha=0.01)
for i in range(3):
    model = train(model, train_data, train_target, iterations=1)
    print("%age of Correct predictions on Test Set: " + str(test(model, test_data, test_target)*100))
