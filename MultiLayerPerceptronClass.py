class MultiLayerPerceptron:
    def __init__(self, independent_features):
        self.learningRate = 0.0001
        self.bias = 1
        self.feature_length = len(independent_features[0])
        self.weight_layer1 = [[random.random() for i in range (feature_length+1)] for j in range (feature_length)]
        self.weight_layer2 = [[random.random() for i in range (feature_length+1)] for j in range (feature_length)]
        self.weight_final = [random.random() for i in range (feature_length+1)]
        self.hx1 = []
        self.hx2 = []

    def hidden_Calculation1(self,input):
        for i in range(len(self.weight_layer1)):
            xi = 0
            for j in range(len(input)):
                xi += input[j] * self.weight_layer1[i][j]
            if (len(self.hx1) < len(input)):
                self.hx1.append(xi)
            else:
                self.hx1[i] = xi
    def hidden_Calculation2(self):
        for i in range(len(self.weight_layer2)):
            xi = 0
            for j in range(len(self.hx1)):
                xi += self.hx1[j] * self.weight_layer2[i][j]
            if (len(self.hx2) < len(self.hx1)):
                self.hx2.append(xi)
            else:
                self.hx2[i] = xi
    def y_Calculation(self):
        xi = 0
        for i in range(len(self.weight_final)):
            xi += self.hx2[i] * self.weight_final[i]
        return xi
    def guess_value(y_val):
        if y_val > 0:
            return 1
        else:
            return 0
    def backpropagation(self, error, inputs):
        for i in range(len(self.weight_layer1)):
            for j in range(self.feature_length):
                self.weight_layer1[i][j] += self.learningRate * error * inputs[j]
        for i in range(len(self.weight_layer2)):
            for j in range(len(self.hx1)):
                self.weight_layer2[i][j] += self.learningRate * error * self.hx1[j]
        for i in range(len(self.weight_final)):
            self.weight_final[i] += self.learningRate * error * self.hx2[i]