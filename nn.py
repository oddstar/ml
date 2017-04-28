import numpy as np
import copy

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
        print("\t%s" % w[i])
    print("bais:")
    for i in range(len(b)):
        print("\t%s" % b[i])
    print("----------------\n")

def print_array(a,title=""):
    print("->%s-" % title)
    for i in range(len(a)):
        print(a[i])
    print("-%s<-" % title)


class NeuralNetwork:
    def __init__(self,layers,weight=[],bias=[]):
        self.activation = tanh
        self.activation_deriv = tanh_deriv


        self.weights = copy.copy(weight)
        self.bias = copy.copy(bias)

        for i in range(len(layers) - 1):
            if(len(weight) == 0):
                self.weights.append(
                    np.random.random((layers[i], layers[i+1]))
                )
            if (len(bias) == 0):
                self.bias.append(
                    np.random.random((layers[i+1]))
                )
        # print_Weight(self.weights,self.bias)

    def fit(self,x,y_,learing_rate=0.5, epochs=100):
        # print("--------------fit---------")
        for i in range(epochs):
            etotal = 0
            for j in range(len(x)):
                a = [x[j]]
                for w in range(len(self.weights)):
                    tmp = self.activation(
                        np.dot(a[-1], self.weights[w]) + self.bias[w]
                    )
                    a.append(tmp.tolist())
                # print_array(a,"output")
                err = np.array(a[-1]) - np.array(y_[j])
                etotal += np.sqrt(np.power(err,2))
                # print(etotal)
                deltas = [err * self.activation_deriv(np.array(a[-1]))]

                for w in range(len(self.weights) - 1,0,-1):
                    # print("idx:%s" % w)
                    deltas.append(
                        np.array(deltas[-1]).dot(np.array(self.weights[w]).T) * tanh_deriv(np.array(a[w]))
                     )
                deltas.reverse()
                # print_array(deltas,"deltas")

                # print("----update weight---")
                for w in range(len(self.weights)-1,-1,-1):
                    layer = np.atleast_2d(a[w])
                    delta = np.atleast_2d(deltas[w])
                    # print("a:%s  deltas:%s  w:%s" % (a[w],deltas[w],self.weights[w]))
                    # self.weights[w] -= learing_rate * np.multiply(deltas[w],dba)
                    self.weights[w] -= learing_rate * layer.T.dot(delta)
                    self.bias[w] -= learing_rate * deltas[w]
                    # print("new_w:%s" % self.weights[w])
                    # print("-----")

                # print_Weight(self.weights,self.bias)
            print("etotal:%s" % (etotal/len(x)))


            # print("-----check------")
            # b = [x[i]]
            # for w in range(len(self.weights)):
            #     tmp = self.activation(
            #         np.dot(b[w], self.weights[w])
            #     )
            #     b.append(tmp.tolist())
            # print("b%s" % b)





if __name__ == '__main__':
    # nn = NeuralNetwork([2,2,2],
    #                    [
    #                        [
    #                            [0.15,0.25],
    #                            [0.2, 0.3]
    #                        ],
    #                        [
    #                            [0.4, 0.5],
    #                            [0.45, 0.55]
    #                        ],
    #                    ],
    #                    [
    #                        [0.35,0.35],
    #                        [0.6, 0.6],
    #                    ])
    # x = [
    #     [0.05,0.1]
    # ]
    # y = [
    #     [0.01,0.99]
    # ]
    nn = NeuralNetwork([2,4,1])
    x = []
    y = []
    for i in range(100):
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