from .utils import np

class Optimizers_: 

    def __init__(self):
        pass

    def step(self, grads, layers):
        raise NotImplementedError("step() function not defined")

class SGD(Optimizers_):

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, grads_W, grads_b, layers):
        for grad_W, grad_b, layer in zip(grads_W, grads_b, layers):
            grad_W = self.lr * grad_W
            grad_b = self.lr * grad_b
            layer.update_params(grad_W, grad_b)

class SGDMomentum(Optimizers_):

    def __init__(self, alpha=0.01, beta=0.9):
        self.alpha = alpha
        self.beta = beta
        self.vW = []
        self.vb = []
    
    def step(self, grads_W, grads_b, layers):
        if len(self.vW) == 0 and len(self.vb) == 0:
            self.vW = [np.zeros_like(grad) for grad in grads_W]
            self.vb = [np.zeros_like(grad) for grad in grads_b]
        
        for i, (grad_W, grad_b, layer) in enumerate(zip(grads_W, grads_b, layers)):
            self.vW[i] = self.beta*self.vW[i] + (1-self.beta)*grad_W
            grad_W = self.alpha * self.vW[i]
            self.vb[i] = self.beta*self.vb[i] + (1-self.beta)*grad_b
            grad_b = self.alpha * self.vb[i]
            layer.update_params(grad_W, grad_b)

class RMSProp(Optimizers_):

    def __init__(self, alpha=0.01, beta=0.9, epsilon=1e-9):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.sW = []
        self.sb = []

    def step(self, grads_W, grads_b, layers):
        if len(self.sW) == 0 and len(self.sb) == 0:
            self.sW = [np.zeros_like(grad) for grad in grads_W]
            self.sb = [np.zeros_like(grad) for grad in grads_b]
        for i, (grad_W, grad_b, layer) in enumerate(zip(grads_W, grads_b, layers)):
            self.sW[i] = self.beta*self.sW[i] + (1-self.beta)*grad_W**2
            grad_W = self.alpha * (grad_W/(np.sqrt(self.sW[i]) + self.epsilon))
            self.sb[i] = self.beta*self.sb[i] + (1-self.beta)*grad_b**2
            grad_b = self.alpha * (grad_b/(np.sqrt(self.sb[i]) + self.epsilon))
            layer.update_params(grad_W, grad_b)

class Adam(Optimizers_):
    
    def __init__(self, alpha=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-9):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.vW = []
        self.sW = []
        self.vb = []
        self.sb = []
        self.t = 1

    def step(self, grads_W, grads_b, layers):
        if len(self.sW) == 0 and len(self.vW) == 0 and len(self.sb) == 0 and len(self.vb) == 0:
            self.vW = [np.zeros_like(grad) for grad in grads_W]
            self.sW = [np.zeros_like(grad) for grad in grads_W]
            self.vb = [np.zeros_like(grad) for grad in grads_b]
            self.sb = [np.zeros_like(grad) for grad in grads_b]
            
        for i, (grad_W, grad_b, layer) in enumerate(zip(grads_W, grads_b, layers)):
            self.vW[i] = (self.beta_1*self.vW[i] + (1-self.beta_1)*grad_W)
            self.sW[i] = (self.beta_2*self.sW[i] + (1-self.beta_2)*grad_W**2)
            v_correct = self.vW[i] / (1-self.beta_1**self.t)
            s_correct = self.sW[i] / (1-self.beta_2**self.t)
            grad_W = self.alpha * (v_correct / (np.sqrt(s_correct) + self.epsilon))
            
            self.vb[i] = (self.beta_1*self.vb[i] + (1-self.beta_1)*grad_b)
            self.sb[i] = (self.beta_2*self.sb[i] + (1-self.beta_2)*grad_b**2)
            v_correct = self.vb[i] / (1-self.beta_1**self.t)
            s_correct = self.sb[i] / (1-self.beta_2**self.t)
            grad_b = self.alpha * (v_correct / (np.sqrt(s_correct) + self.epsilon))
            
            layer.update_params(grad_W, grad_b)
        self.t += 1