import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)


class NeuralNetwork:
    def __init__(self,layers):
        self.activation = tanh
        self.activation_deriv = tanh_deriv

        self.weights = []
        self.bias = []

        for i in range(len(layers) - 1):
            self.weights.append(
                np.random.random((layers[i],layers[i+1]))
            )
            self.bias.append(
                np.random.random((layers[i],layers[i+1]))
            )
        for i in range(len(self.weights)):
            print(self.weights[i])
            print("")

    def fit(self,x,y_,learing_rate=0.02,epochs=1000):
        print("--------------fit---------")
        for i in range(epochs):
            a = [x[i]]
            for w in range(len(self.weights)):
                tmp = self.activation(
                    np.dot(a[w], self.weights[w])
                )
                a.append(tmp.tolist())
            print(a)
            err = np.array(y_[i]) - np.array(a[-1])
            deltas = [err * self.activation_deriv(a[-1])]
            print("deltas:%s" % deltas)

            for w in range(len(self.weights) - 1,0,-1):
                print("++++++++++++")
                print("idx:%s" % w)
                print("del:%s  w:%s   a:%s" % (deltas[-1],self.weights[w].T,a[w]))
                deltas.append(
                    deltas[-1].dot(self.weights[w].T) * self.activation_deriv(a[w])
                )
                print("deltas:%s" % deltas[len(deltas) - 1])
            deltas.reverse()
            print(deltas)

            print("----update weight---")
            for w in range(len(self.weights)):
                print("a:%s  deltas:%s  w:%s" % (a[w],deltas[w],self.weights[w]))
                self.weights[w] += learing_rate * np.array(a[w+1]).T.dot(deltas[w])
            print("new_weight:%s" % self.weights)

            print("-----check------")
            b = [x[i]]
            for w in range(len(self.weights)):
                tmp = self.activation(
                    np.dot(b[w], self.weights[w])
                )
                b.append(tmp.tolist())
            print("b%s" % b)





if __name__ == '__main__':
    nn = NeuralNetwork([2,3,1])
    x = []
    y = []
    for i in range(10000):
        a = np.random.random()
        b = np.random.random()
        x.append([a,b])
        y.append(a+b)
    nn.fit(x,y)
    # nn.fit(
    #     [
    #         [0.3,0.6],
    #         [0.2,0.33],
    #         [0.4,0.2],
    #         [0.2,0.4]
    #     ],
    #     [
    #         0.9,
    #         0.5,
    #         0.6,
    #         0.6
    #     ]
    # )
