import numpy as np
import time
from tqdm import tqdm

from .activations import *
from ._grads import *
from .Initializers import *
from .Layers import *
from .Optimizers import *
from .Losses import *

class Model:
    def __init__(self, optimizer:object, layers:list, loss_func:object=CrossEntropy()):
        """
        Deep neural network architecture.
        Parameters
        ----------
        optimizer: (object) optimizer object uses to optimize the loss.
        layers: (list) a list of sequential layers. For neural network, it should have [Dense, Activation, BatchnormLayer, Dropout]
        loss_func: (object) the type of loss function we want to optimize. 
        """
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.layers = layers
        if isinstance(self.layers[-1], Activation):
            self.layers[-1].last = True

    def _forward(self, train_X, prediction=False):
        """
        NN forward propagation level.
        Parameters
        ----------
        train_X: training dataset X.
                shape = (N, D)
        prediction: whether this forward pass is prediction stage or training stage.
        Returns
        -------
        Probability distribution of softmax at the last layer.
            shape = (N, C)
        """
        inputs = train_X
        layers = self.layers
            
        for layer in layers:
            if isinstance(layer, Dropout):
                inputs = layer.forward(inputs, prediction=prediction)
                continue
            inputs = layer.forward(inputs)
        output = inputs
        return output
    
    def __call__(self, X, prediction=False):
        return self._forward(X, prediction)

    def _update_params(self, grads):
        self.optimizer.step(grads)

    def backward(self, Y, Y_hat, X):
        """
        Parameters
        ----------
        Y: one-hot encoding label.
            shape = (N, C).
        Y_hat: output values of forward propagation NN.
            shape = (N, C).
        X: training dataset.
            shape = (N, D).
        """
        if not hasattr(self, "output_layers"):
            self.learnable_layers = [layer for layer in self.layers if isinstance(layer, LearnableLayer)]
            self.learnable_layers = self.learnable_layers[::-1]

        grads = []
        
        dCost = self.loss_func.backward(Y_hat, Y)
        dA_prev, dW = dCost, None
        
        for i in range(len(self.layers)-1, 0, -1):
            if isinstance(self.layers[i], LearnableLayer):
                dA_prev, dW, db = self.layers[i].backward(dA_prev, self.layers[i-1])
                dlayer = {'layer': self.layers[i], 'dWs': dW, 'dbs' : db}
                grads.append(dlayer)
            else:
                dA_prev = self.layers[i].backward(dA_prev, self.layers[i-1])

        self._update_params(grads)

    def convertOneHot(self, input_):
        result = np.zeros((input_.size, input_.max()+1))
        result[np.arange(input_.size),input_] = 1
        return result


    def predict(self, test_X):
        """
        Predict function.
        """
        y_hat = self._forward(test_X, prediction=True)
        return np.argmax(y_hat, axis=1)
    

    def cal_accuracy(self, Y, pred):
        m = Y.shape[0]
        if len(Y.shape) > 1:
            Y = np.argmax(Y, axis=1).reshape(m)
        else:
            Y = Y.reshape(m)
        return len(pred[Y == pred]) / len(pred)
        
    
    def fit(self, X_train, y_train, validation, batch_size, epochs):
        m = X_train.shape[0]
        X_val, y_val = validation
        train_losses, val_losses = [],[]
        train_accs, val_accs = [],[]
        
        for e in range(epochs):
            indices = np.random.permutation(m)
            X_train = X_train[indices]
            y_train = y_train[indices]
            epoch_loss, val_loss = 0.0, 0.0
            num_batches, val_batches = 0, 0
            pbar = tqdm(range(0, X_train.shape[0], batch_size))
            
            for it in pbar:
                X_batch = X_train[it:it+batch_size]
                y_batch = y_train[it:it+batch_size]
                
                y_hat = self._forward(X_batch, prediction=False)
                batch_loss = self.loss_func(y_hat, y_batch)
                self.backward(y_batch, y_hat, X_batch)

                epoch_loss += batch_loss
                num_batches += 1
                pbar.set_description("Epoch " + str(e+1) + " - Loss: %.5f" % (epoch_loss/num_batches))
                
            for it in range(0, X_val.shape[0], batch_size):
                X_batch = X_val[it:it+batch_size]
                y_batch = y_val[it:it+batch_size]
                
                y_val_hat = self._forward(X_batch)
                batch_val_loss = self.loss_func(y_val_hat, y_batch)
                val_loss += batch_val_loss
                val_batches += 1
            
            train_losses.append(epoch_loss/num_batches)
            val_losses.append(val_loss/val_batches)
            
            train_acc = self.cal_accuracy(y_train, self.predict(X_train))
            val_acc = self.cal_accuracy(y_val, self.predict(X_val))
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            pbar.close()
            print("Loss at epoch %d: %.5f - Train accuracy: %.5f - Validation loss: %.5f - accuracy: %.5f" % (e+1, epoch_loss/num_batches, train_acc, val_loss/val_batches, val_acc))
            time.sleep(1)
        return train_losses, val_losses, train_accs, val_accs