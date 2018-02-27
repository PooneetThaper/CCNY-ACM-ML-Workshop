import numpy as np

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

y = np.array([[0,0,0],
             [0,1,1],
             [0,1,1],
             [1,1,0]])
weights=[]

def sigmoid(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

def train(n,iter_n):
    for j in range(iter_n):
        l=[]
        l.append(X)
        g=1
        while g<(n+2):
            l.append(sigmoid(np.dot(l[g-1], weights[g-1])))
            g+=1

        # Back propagation of errors using the chain rule.
        errors = []
        deltas = []

        #Top level error and delta
        top_error = y - l[n+1]
        errors.append(top_error)
        top_delta = top_error*sigmoid(l[n+1],deriv=True)
        deltas.append(top_error)
        #Deeper level error and delta
        for k in range(n):
            e=deltas[k].dot(weights[n-k].T)
            errors.append(e)
            d=e*sigmoid(l[n-k],deriv=True)
            deltas.append(d)

        #Original for n=3
        #l4_error = y - l[4]
        #l4_delta = l4*sigmoid(l[4],deriv=True)
        #l3_error = l4_delta.dot(weights[3].T)
        #l3_delta = l3_error*sigmoid(l[3],deriv=True)
        #l2_error = l3_delta.dot(weights[2].T)
        #l2_delta = l2_error*sigmoid(l[2], deriv=True)
        #l1_error = l2_delta.dot(weights[1].T)
        #l1_delta = l1_error * sigmoid(l[1],deriv=True)

        if(j % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output.
            print(j,":Error: ",str(np.mean(np.abs(top_error))))

        #update weights (no learning rate term)
        for k in range(n+1):
            weights[k] += np.transpose(l[k]).dot(deltas[n-k])/2

        #Original for n=3
        #weights[3] += l[3].T.dot(l4_delta)/2
        #weights[2] += l[2].T.dot(l3_delta)/2
        #weights[1] += l[1].T.dot(l2_delta)/2
        #weights[0] += l[0].T.dot(l1_delta)/2


def build(numIn,numOut,numHiddenLayers,numNeuronsHidden):
    last=numIn
    np.random.seed(1)
    for i in range(numHiddenLayers):
        weights.append(2*np.random.random((last,numNeuronsHidden))-1)
        last = numNeuronsHidden
    weights.append(2*np.random.random((last,numOut))-1)

def test(n):
    l=[]
    l.append(X)
    g=1
    while g<(n+2):
        l.append(sigmoid(np.dot(l[g-1], weights[g-1])))
        g+=1
    print(l[n+1])

def main():
    # Number of inputs (2 plus a bias in this case)
    numInputs=3
    # Number of outputs (3 for AND, OR, and XOR functions)
    numOutputs=3
    # Number of hidden layers
    numHiddenLayers=1
    # Number of nodes in each hidden layer
    numNeuronsHidden=3

    print(X)
    print(y)
    build(numInputs,numOutputs,numHiddenLayers,numNeuronsHidden)
    for layer in weights:
        print(weights)
    train(numHiddenLayers,100000)
    test(numHiddenLayers)


if __name__ == '__main__':
    main()
