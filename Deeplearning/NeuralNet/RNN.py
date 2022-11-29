import numpy as np

from .activations import *
from ._grads import *
from .Initializers import initialization_mapping
from .Layers import LearnableLayer

class SimpleRNN(LearnableLayer):

    def __init__(self, units, weight_init="std",
                 activation=None, recurrent_activation='tanh', bptt_trunc = 5,
                 use_bias=True, return_sequences=False, return_state=False):
        """
        The Recurrent Neural Network.
        Parameters
        ----------
        units: (integer) number of neurons in the units.     
        weight_init: (string) either `he_normal`, `xavier_normal`, `he_uniform`, `glorot_uniform` or standard normal distribution.
        """
        assert weight_init in initialization_mapping.keys(),\
                "Unknow weight initialization type."
        assert activation in ["swish", "sigmoid", "tanh", "relu", swish, sigmoid, tanh, relu, None],\
                "Unknown activation function: " + str(activation)
        assert recurrent_activation in ["swish", "sigmoid", "tanh", "relu", swish, sigmoid, tanh, relu],\
                "Unknown activation function: " + str(activation)
        
        self.n_units = units
        self.weight_init = weight_init
        self.output = None
        self.paramater = { 'W_xh' : None, 'W_hh' : None, 'W_ho' : None}

        if isinstance(activation, str):
            self.activation = activation_mapping[activation]
        else:
            self.activation = activation
        
        if isinstance(recurrent_activation, str):
            self.recurrent_activation = activation_mapping[recurrent_activation]
        else:
            self.recurrent_activation = recurrent_activation
        
        self.use_bias = use_bias
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.bptt_trunc = bptt_trunc

    def forward(self, X):
        """
        """
        self.layer_input = X
        self.batch_size, self.timesteps, self.input_dim = X.shape

        if self.paramater['W_xh'] is None:
            self.paramater['W_xh'] = initialization_mapping[self.weight_init](weight_shape=(self.n_units, self.input_dim))
        if self.paramater['W_hh'] is None:
            self.paramater['W_hh'] = initialization_mapping[self.weight_init](weight_shape=(self.n_units, self.n_units))
        if self.paramater['W_ho'] is None:
            self.paramater['W_ho'] = initialization_mapping[self.weight_init](weight_shape=((self.input_dim, self.n_units)))

        # Save these values for use in backprop.
        self.state_input = np.zeros((self.batch_size, self.timesteps, self.n_units))
        self.states = np.zeros((self.batch_size, self.timesteps+1, self.n_units))
        self.outputs = np.zeros((self.batch_size, self.timesteps, self.input_dim))

        # Set last time step to zero for calculation of the state_input at time step zero
        self.states[:, -1] = np.zeros((self.batch_size, self.n_units))
        for t in range(self.timesteps):
            # Input to state_t is the current input and output of previous states
            self.state_input[:, t]= X[:, t].dot(self.paramater['W_xh'].T) + self.states[:, t-1].dot(self.paramater['W_hh'].T)
            self.states[:, t] = self.recurrent_activation(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.paramater['W_ho'].T)
            if self.activation is not None:
                self.outputs[:, t] = self.activation(self.states[:, t].dot(self.paramater['W_ho'].T))
        
        if(self.return_sequences):
            self.output = self.outputs
            return self.outputs
        else:
            self.output = self.outputs[:,-1,:]
            return self.outputs[:,-1,:]

        # if(self.return_state):
        #     return_outs.append(self.states[:,-1,:])        
        

    def backward(self, d_prev, prev_layer):
        """
        """
        if len(d_prev.shape) == 2:
            _,h_dim = d_prev.shape
            zeros = np.zeros((self.batch_size,self.timesteps,h_dim))
            zeros[:,-1,:] = d_prev
            d_prev = zeros 
        
        # _, timesteps, _ = d_prev.shape
        #U = Wxh, W = Whh, V = Who

        # Variables where we save the accumulated gradient w.r.t each parameter
        grad_Wxh = np.zeros_like(self.paramater['W_xh'])
        grad_Whh = np.zeros_like(self.paramater['W_hh'])
        grad_Who = np.zeros_like(self.paramater['W_ho'])
        # The gradient w.r.t the layer input.
        # Will be passed on to the previous layer in the network
        d_prev_next = np.zeros_like(d_prev)

        # Back Propagation Through Time
        for t in reversed(range(self.timesteps)):
            # Update gradient w.r.t V at time step t
            grad_Who += d_prev[:, t].T.dot(self.states[:, t])
            # Calculate the gradient w.r.t the state input
            grad_wrt_state = d_prev[:, t].dot(self.paramater['W_ho'])
            if self.activation is not None:
                grad_wrt_state *= eval(self.activation.__name__+'_grad')(self.state_input[:, t])
            # Gradient w.r.t the layer input
            d_prev_next[:, t] = grad_wrt_state.dot(self.paramater['W_xh'])
            # Update gradient w.r.t W and U by backprop. from time step t for at most
            # self.bptt_trunc number of time steps
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
                grad_Wxh += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_Whh += grad_wrt_state.T.dot(self.states[:, t_-1])
                # Calculate gradient w.r.t previous state
                grad_wrt_state = grad_wrt_state.dot(self.paramater['W_hh'])
                if self.recurrent_activation is not None:
                    grad_wrt_state *= eval(self.recurrent_activation.__name__+'_grad')(self.state_input[:, t_-1])
        
        dWs = {'W_xh': grad_Wxh, 'W_hh': grad_Whh, 'W_ho': grad_Who}
        # Update weights
        return d_prev_next, dWs, None
