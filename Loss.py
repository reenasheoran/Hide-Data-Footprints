
class CrossEntropyLoss(object):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        return input.cross_entropy(target)

class MSELoss(object):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        dif = input - target
        return (dif * dif).sum(0)


