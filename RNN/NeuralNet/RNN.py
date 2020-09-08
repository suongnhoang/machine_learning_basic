from .utils import np
from .Activations import *
from ._grads import *
from .Layers import Layer, LearnableLayer, initialization_mapping, activation_mapping

class RNN(LearnableLayer):

    def __init__(self, units, weight_init="std",
                 activation='tanh', recurrent_activation='sigmoid', 
                 use_bias=True, return_sequences=False, return_state=False):
        """
        The Recurrent Neural Network.
        Parameters
        ----------
        units: (integer) number of neurons in the units.     
        weight_init: (string) either `he_normal`, `xavier_normal`, `he_uniform`, `glorot_uniform` or standard normal distribution.
        """
        assert weight_init in ["std", "glorot_normal", "he_normal", "he_uniform", "glorot_uniform"],\
                "Unknow weight initialization type."
        assert activation in ["swish", "sigmoid", "tanh", "relu", swish, sigmoid, tanh, relu],\
                "Unknown activation function: " + str(activation)
        assert recurrent_activation in ["swish", "sigmoid", "tanh", "relu", swish, sigmoid, tanh, relu],\
                "Unknown activation function: " + str(activation)
        
        self.n_units = units
        self.weight_init = weight_init
        self.output = None
        self.W_xh = None # Weight of the previous state
        self.W_hh = None # Weight of the output
        self.W_ho = None # Weight of the input
        
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

    def forward(self, X):
        """
        """
        self.layer_input = X
        batch_size, timesteps, input_dim = X.shape

        # Save these values for use in backprop.
        outputs = []
        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps+1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))

        # Set last time step to zero for calculation of the state_input at time step zero
        self.states[:, -1] = np.zeros((batch_size, self.n_units))
        for t in range(timesteps):
            # Input to state_t is the current input and output of previous states
            self.state_input[:, t] = X[:, t].dot(self.W_xh.T) + self.states[:, t-1].dot(self.W_hh.T)
            self.states[:, t] = tanh(self.state_input[:, t])
            self.outputs[:, t] = self.activation(self.states[:, t].dot(self.W_ho.T))
        
        if(self.return_sequences):
            outputs.append(self.outputs)

        outputs.append(self.outputs[:, -1])

        if(self.return_state):
            outputs.append(self.states[:, -1])
        
        return outputs

    def backward(self, accum_grad, prev_layer):
        """
        """
        _, timesteps, _ = accum_grad.shape

        # Variables where we save the accumulated gradient w.r.t each parameter
        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)
        # The gradient w.r.t the layer input.
        # Will be passed on to the previous layer in the network
        accum_grad_next = np.zeros_like(accum_grad)

        # Back Propagation Through Time
        for t in reversed(range(timesteps)):
            # Update gradient w.r.t V at time step t
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])
            # Calculate the gradient w.r.t the state input
            grad_wrt_state = accum_grad[:, t].dot(self.V) * self.activation.gradient(self.state_input[:, t])
            # Gradient w.r.t the layer input
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
            # Update gradient w.r.t W and U by backprop. from time step t for at most
            # self.bptt_trunc number of time steps
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_-1])
                # Calculate gradient w.r.t previous state
                grad_wrt_state = grad_wrt_state.dot(self.W) * self.activation.gradient(self.state_input[:, t_-1])

        # Update weights
        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)

        return accum_grad_next


        def update_params(self, grad_U, grad_V, grad_W):
            self.W = self.W - grad_W
            self.U = self.U - grad_U
            self.V = self.V - grad_V