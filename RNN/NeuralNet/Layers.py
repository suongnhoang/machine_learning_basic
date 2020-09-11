from .utils import *
from .Activations import *
from ._grads import *
from .Initializers import *


class Layer(object):

    def forward(self, X):
        raise NotImplementedError("forward() function not defined")

    def backward(self):
        raise NotImplementedError("backward() function not defined")

class LearnableLayer(object):

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

    def __init__(self, num_neurons, weight_init="std"):
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
    
        if self.paramater['bias'] is None:
            self.paramater['bias']=initialization_mapping[self.weight_init](weight_shape=(1, self.num_neurons))
        
        self.output = np.dot(inputs,self.paramater['W'])+self.paramater['bias']
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
        dW = np.dot(prev_layer.output.T, d_prev)
        db = np.sum(d_prev,axis=0)
        d_prev = self.backward_layer(d_prev, None)
        return d_prev, {'W':dW}, {'bias':db}


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
        assert activation in ["swish", "sigmoid", "tanh", "relu", "softmax"], "Unknown activation function: " + str(activation)
        self.activation = activation
        self.last = False

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
