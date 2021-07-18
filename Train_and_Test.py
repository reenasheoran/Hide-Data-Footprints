import sys
from Gradient import SGD
from Loss import MSELoss
from Tensor import Tensor
from DataPrep import w2i

def train(model, input_data, target_data, batch_size=500, iterations=1):
    bs = batch_size
    criterion = MSELoss()
    optim = SGD(parameters=model.get_parameters(), alpha=0.01)
    
    n_batches = int(len(input_data) / batch_size)
    for iter in range(iterations):
        iter_loss = 0
        for b_i in range(n_batches):

            # padding token should stay at 0
            model.weight.data[w2i['<unk>']] *= 0 
            input = Tensor(input_data[b_i*bs:(b_i+1)*bs], autograd=True)
            target = Tensor(target_data[b_i*bs:(b_i+1)*bs], autograd=True)

            pred = model.forward(input).sum(1).sigmoid()
            loss = criterion.forward(pred,target)
            loss.backward()
            optim.step()

            iter_loss += loss.data[0] / bs

            sys.stdout.write("\r\tLoss:" + str(iter_loss / (b_i+1)))
        print()
    return model

def test(model, test_input, test_output):
    
    model.weight.data[w2i['<unk>']] *= 0 
    
    input = Tensor(test_input, autograd=True)
    target = Tensor(test_output, autograd=True)

    pred = model.forward(input).sum(1).sigmoid()
    return ((pred.data > 0.5) == target.data).mean()

