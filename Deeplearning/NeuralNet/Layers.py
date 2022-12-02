import numpy as np
from abc import ABC

from .activations import *
from ._grads import *
from .Initializers import *


class Layer(ABC):
    def __init__(self):
        pass

    def forward(self, X):
        raise NotImplementedError("forward() function not defined")

    def backward(self):
        raise NotImplementedError("backward() function not defined")


class LearnableLayer(ABC):
    def __init__(self):
        pass

    def forward(self, X):
        raise NotImplementedError("forward() function not defined")

    def backward_layer(self):
        pass

    def backward(self):
        raise NotImplementedError("backward() function not defined")

    def update_params(self, grads_update):
        for weight_name in grads_update:
            self.paramater[weight_name]=self.paramater[weight_name]-grads_update[weight_name]


class Input(Layer):
    def __init__(self, return_dX=False):
        self.return_dX = return_dX
        self.output = None
        super().__init__()

    def forward(self, X):
        self.output = X
        return self.output

    def backward(self, d_prev, weights_prev):
        """
        d_prev: gradient of J respect to A[l+1] of the previous layer according backward direction.
        weights_prev: the weights of previous layer according backward direction.
        """
        if self.return_dX:
            return np.dot(d_prev, weights_prev.T)
        return None


class Dense(LearnableLayer):

    def __init__(self, num_neurons, weight_init="std", use_bias = True):
        """
        The fully connected layer.
        Parameters
        ----------
        num_neurons: (integer) number of neurons in the layer.     
        weight_init: (string) either `he_normal`, `xavier_normal`, `he_uniform`, `glorot_uniform` or standard normal distribution.
        """
        assert weight_init in ["std", "glorot_normal", "he_normal", "he_uniform", "glorot_uniform"],\
                "Unknow weight initialization type."
        self.num_neurons = num_neurons
        self.weight_init = weight_init
        self.output = None
        self.paramater = {'W':None,'bias':None}
        self.use_bias = use_bias
        super().__init__()

    def forward(self, inputs):
        """
        Layer forward level. 
        Parameters
        ----------
        inputs: inputs of the current layer. This is equivalent to the output of the previous layer.
        Returns
        -------
        output: Output value LINEAR of the current layer.
        """
        if self.paramater['W'] is None:
            self.paramater['W']=initialization_mapping[self.weight_init](weight_shape=(inputs.shape[1], self.num_neurons))
    
        if self.paramater['bias'] is None and self.use_bias:
            self.paramater['bias']=initialization_mapping[self.weight_init](weight_shape=(1, self.num_neurons))
        
        self.output = np.dot(inputs,self.paramater['W'])
        
        if self.use_bias:
            self.output += self.paramater['bias']
        
        return self.output

    def backward_layer(self, d_prev, _):
        """
        Compute gradient w.r.t X only.
        """
        d_prev = np.dot(d_prev, self.paramater['W'].T)
        return d_prev

    def backward(self, d_prev, prev_layer):
        """
        Layer backward level. Compute gradient respect to W and update it.
        Also compute gradient respect to X for computing gradient of previous
        layers as the forward direction [l-1].
        Parameters
        ----------
        d_prev: gradient of J respect to A[l+1] of the previous layer according backward direction.
        prev_layer: previous layer according forward direction.
        Returns
        -------
        d_prev: gradient of J respect to A[l] at the current layer.
        """
        dWs, dbs = None, None
        dW = np.dot(prev_layer.output.T, d_prev)
        dWs = {'W':dW}
        if self.use_bias:
            db = np.sum(d_prev,axis=0)
            dbs = {'bias':db}        
        d_prev = self.backward_layer(d_prev, None)
        return d_prev, dWs, dbs


class Dropout(Layer):
    """
    Refer to the paper: 
        http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    """
    def __init__(self, keep_prob):
        """
        keep_prob: (float) probability to keep neurons in network, use for dropout technique.
        """
        assert 0.0 < keep_prob < 1.0, "keep_prob must be in range [0, 1]."
        self.keep_prob = keep_prob
        super().__init__()

    def forward(self, X, prediction=False):
        """
        Drop neurons random uniformly.
        """
        if prediction:
            self.output = X * self.keep_prob
            return self.output
        
        self.mask = np.random.uniform(size=X.shape) < self.keep_prob
        self.output = X * self.mask
        return self.output

    def backward(self, d_prev, _):
        """
        Flow gradient of previous layer [l+1] according backward direction through dropout layer.
        """
        return d_prev * self.mask


class Activation(Layer):

    def __init__(self, activation):
        """
        activation: (string) available activation functions. Must be in [sigmoid, tanh,
                                relu, softmax]. Softmax activation must be at the last layer.
        
        """
        assert activation in activation_mapping.keys(), f"Unknown activation function: {activation}"
        self.activation = activation
        self.last = False
        super().__init__()

    def forward(self, X):
        """
        Activation layer forward propgation.
        """
        self.output = eval(self.activation)(X)
        self.input = X
        return self.output

    def backward(self, d_prev, _):
        """
        Activation layer backward propagation.
        Parameters
        ---------- 
        d_prev: gradient of J respect to A[l+1] of the previous layer according backward direction.
        
        Returns
        -------
        Gradient of J respect to type of activations (sigmoid, tanh, relu) in this layer `l`.
        """
        if self.last:
            # return previous derivatives of loss, because we computed derivatives of softmax with CE-loss already.
            # ref: https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
            return d_prev
        d_prev = d_prev * eval(self.activation + "_grad")(self.input)
        return d_prev


class GNN(Layer):
    def __init__(self,T,D):
        """
        Class initialization, takes hyper param T and D
        Args:
            T : aggregation step
            D : feature vector size
        class_params:
            W, A, b : parameters of the network. W and A is
                    initialized using normal distribution
                    with sigma = 0.4 and mean = 0
            dLdW, dLdA, dLdb : gradient of the parameters
        """
        sigma = 0.4
        self.T, self.D = T, D

        self.paramater = {'W':sigma * np.random.randn(D,D), 'A': sigma * np.random.randn(D),'bias':0}

        self.dLdW = np.zeros((D,D))
        self.dLdA = np.zeros((D))
        self.dLdb = 0
        super().__init__()
        
    def aggregation(self, W, X, adj):
        """
        Function to calculate aggregation 2, here transpose 
        is used to easily calculating the aggregation using 
        dot product
        Args :
            W : D x D weight matrix 
            a : Output of aggregation1 
        Return :
            x : W . a
        """
        a = np.dot(adj, X)
        x = np.dot(W,np.transpose(a))
        x = np.transpose(x)
        return x

    def forward(self, nnodes, adj):
        """
        forward method to calculate forward propagation of the nets
        Args :
            nnodes  : number of nodes in the batch
            adj     : adjacency matrix
            W       : parameter matrix W
            A       : parameter vector A
            b       : bias b
        Return : 
            slist       : vector of predictor value 
            output list : vector of predicted class`
        """
        slist, outputlist, X = [], [], []       
        # feature vector definition
        feat =  np.zeros(self.D)
        feat[0] = 1
        
        self.tempnnodes, self.tempadj = nnodes, adj

        for i in range(adj.shape[0]):
            X.append(np.tile(feat,[nnodes[i],1]))
            # Message passing
            for _ in range(self.T):
                X[i] = relu(self.aggregation(self.paramater['W'], X[i], adj[i]))
            hG = np.sum(X[i], axis=0) #sum all node feature vectors
            s = np.dot(hG, self.paramater['A']) + self.paramater['bias'] # Predictor function 
            slist.append(s)
            
            #read_out stage
            p = sigmoid(s)
            output = np.where((p>0.5),1,0)
            outputlist.append(int(output))
        
        return slist, outputlist

    def backward(self, loss, y, epsilon):
        """
        Backpropagation function to calculate and update 
        the gradient of the neural network
        Args :
            loss    : loss vector
            y       : true class label
            epsilon : small pertubation value for numerical 
                      differentiation 
        """
        tempdLdW = np.zeros((self.D, self.D))
        tempdLdA = np.zeros((self.D))
        tempdLdb = 0
        batchsize = len(loss)
        
        for i in range(self.D):
            for j in range (self.D):
                deltaW = np.zeros((self.D, self.D))
                deltaW[i,j]=epsilon
                Wepsilon = self.paramater['W'] + deltaW
                sep,_ = self.forward(self.tempnnodes, self.tempadj, W=Wepsilon)
                lossep = self.loss(sep, y)
                for k in range(batchsize):
                    tempdLdW[i,j] += (lossep[k] - loss[k])/epsilon
                tempdLdW[i,j] = tempdLdW[i,j]/batchsize

        for i in range(self.D):
            deltaA = np.zeros((self.D))
            deltaA[i] = epsilon
            Aepsilon = self.paramater['A'] + deltaA
            sep,_ = self.forward(self.tempnnodes, self.tempadj, A=Aepsilon)
            lossep = self.loss(sep, y)   
            for j in range(batchsize):
                tempdLdA[i] += (lossep[j] - loss[j])/epsilon
            tempdLdA[i] = tempdLdA[i]/batchsize

        bepsilon = self.paramater['bias'] + epsilon
        sep,_ = self.forward(self.tempnnodes, self.tempadj,b=bepsilon)
        lossep = self.loss(sep, y) 
        for i in range(batchsize):
            tempdLdb += (lossep[i] - loss[i])/epsilon
        tempdLdb = tempdLdb/batchsize

        self.dLdW = tempdLdW
        self.dLdA = tempdLdA
        self.dLdb = tempdLdb

    def loss(self, s, y):
        """
        loss function
        Args :
            s   : vector of predictor values
            y   : vector of true class labels
        Return :
            losslist : vector of loss values
        """
        losslist = []
        for i in range(len(s)):
            if np.exp(s[i]) > np.finfo(type(np.exp(s[i]))).max:
                loss = y[i]*np.log(1+np.exp(-s[i])) + (1-y[i]) * s[i] #avoid overflow
            else :
                loss = y[i]*np.log(1+np.exp(-s[i])) + (1-y[i]) * np.log(1+np.exp(s[i]))
            losslist.append(loss)
        return losslist
            
    def updateweight(self, W, A, b):
        """
        update weight function
        Args :
            W: parameter matrix W
            A: parameter vector A
            b: bias b
        """
        self.paramater['W'] = W
        self.paramater['A'] = A
        self.paramater['bias'] = b