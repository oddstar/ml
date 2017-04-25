import numpy as np


'''
   激励函数 
'''
def tanh(x):
    return 1 / (1 + np.power(np.e,-1 * x))

def tanh_deriv(x):
    return x*(1-x)



def print_Weight(w,b):
    print("----------------")
    print("weight:")
    for i in range(len(w)):
        print(w[i])
    print("bais:")
    for i in range(len(b)):
        print(b[i])
    print("---------bias-------")




class NeuralNetwork:
    def __init__(self,layers,weight=[],bias=[]):
        self.activation = tanh
        self.activation_deriv = tanh_deriv

        self.weights = weight
        self.bias = bias

        for i in range(len(layers) - 1):
            if(len(weight) == 0):
                self.weights.append(
                    np.random.random((layers[i], layers[i+1]))
                )
            if (len(bias) == 0):
                self.bias.append(
                    np.random.random((layers[i+1]))
                )
        print_Weight(self.weights,self.bias)

    def fit(self,x,y_,learing_rate=0.02, epochs=1):
        print("--------------fit---------")
        for i in range(epochs):
            a = [x[i]]
            for w in range(len(self.weights)):
                tmp = self.activation(
                    np.dot(a[w], self.weights[w]) + self.bias[w]
                )
                a.append(tmp.tolist())
            print(a)
            # err = np.power(np.array(y_[i]) - np.array(a[-1]),2) * 1/2
            err = np.array(a[-1]) - np.array(y_[i])
            print(err)

            # deltas = [np.multiply(err,self.activation_deriv(np.array(a[-1])))]
            deltas = [err]
            print(deltas)
            # print("deltas:%s" % deltas)
            #
            for w in range(len(self.weights),0,-1):
                idx = w - 1
                print("++++++++++++")
                print("idx:%s" % idx)
                print("del:%s  w:%s   a%s:%s  a%s:%s" % (deltas[-1],np.array(self.weights[idx]).T,idx,a[idx],w,a[w]))
                deltas.append(
                    # deltas[-1].dot(self.weights[w].T) * self.activation_deriv(a[w])
                    deltas[-1].dot(np.array(a[idx]).T) * self.activation_deriv(np.array(a[w]))
                )
                print("deltas:%s" % deltas[len(deltas) - 1])
            deltas.reverse()
            # print(deltas)
            #
            # print("----update weight---")
            # for w in range(len(self.weights)):
            #     print("a:%s  deltas:%s  w:%s" % (a[w],deltas[w],self.weights[w]))
            #     self.weights[w] += learing_rate * np.array(a[w+1]).T.dot(deltas[w])
            # print("new_weight:%s" % self.weights)
            #
            # print("-----check------")
            # b = [x[i]]
            # for w in range(len(self.weights)):
            #     tmp = self.activation(
            #         np.dot(b[w], self.weights[w])
            #     )
            #     b.append(tmp.tolist())
            # print("b%s" % b)





if __name__ == '__main__':
    nn = NeuralNetwork([2,2,2],
                       [
                           [
                               [0.15,0.25],
                               [0.2, 0.3]
                           ],
                           [
                               [0.4, 0.5],
                               [0.45, 0.55]
                           ],
                       ],
                       [
                           [0.35,0.35],
                           [0.6, 0.6],
                       ])
    x = [
        [0.05,0.1]
    ]
    y = [
        [0.01,0.99]
    ]
    # for i in range(1):
    #     a = np.random.random()
    #     b = np.random.random()
    #     x.append([a,b])
    #     y.append(a+b)
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
