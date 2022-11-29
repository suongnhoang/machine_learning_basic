from .activations import *

def sigmoid_grad(z):
    """
    Sigmoid derivative.
        g'(z) = g(z)(1-g(z))
    """
    return sigmoid(z)*(1-sigmoid(z))

def swish_grad(z):
    """
    Swish derivative.
        g'(z) = g(z) + sigmoid(z)(1-g(z))
    """
    return swish(z) + sigmoid(z)*(1-swish(z))

def tanh_grad(z):
    """
    Tanh derivative.
        g'(z) = 1 - g^2(z).
    """
    return 1 - tanh(z)**2

def relu_grad(z):
    """
    Relu derivative.
        g'(z) = 0 if g(z) <= 0
        g'(z) = 1 if g(z) > 0
    """
    return 1*(relu(z) > 0)