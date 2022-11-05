import random
def Preceptron(learningRate = 0.0001, bias = 1):
    weights = []
    
    def getInitialWeights(feature_length):
        for i in range(feature_length+1):
            weights.append(random.random())
        return weights
    
    def activate(inputs):
        sum = 0
        for i in range(len(inputs)):
            sum += inputs[i] * weights[i]
        if (sum > 0):
            return 1
        else:
            return 0