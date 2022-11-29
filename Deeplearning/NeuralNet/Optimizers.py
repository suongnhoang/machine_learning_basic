import numpy as np


class Optimizers_: 

    def __init__(self):
        pass

    def step(self, grads):
        """
        paramaters:
            - grads
                + type: List
                + describe: a list to held every single dict, which contains information about gradients 
                            of all weights and bias (more than 2 weights or not)

                            [
                                {
                                    'layer' : __layer_object__,
                                    'dWs' : {'Whh': numpy.array, 'Wxh': numpy.array,...},
                                    'dbs' : {'bhh': numpy.array, 'bxh': numpy.array,...}
                                },
                                ...
                            ] 
        """
        raise NotImplementedError("step() function not defined")


class SGD(Optimizers_):

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, grads):
        for layer_grad in grads:
            grads_update = {}
            for weight_name in layer_grad['dWs']:
                grads_update[weight_name] = self.lr * layer_grad['dWs'][weight_name]
            for bias_name in layer_grad['dbs']:
                grads_update[bias_name] = self.lr * layer_grad['dbs'][bias_name]
            layer_grad['layer'].update_params(grads_update)


class SGDMomentum(Optimizers_):

    def __init__(self, alpha=0.01, beta=0.9):
        self.alpha = alpha
        self.beta = beta
        self.paramaters = []
    
    def step(self, grads):
        if len(self.paramaters) == 0:
            for layer_grad in grads:
                v = {'Ws':{},'bs':{}}

                for W_name in layer_grad['dWs']:
                    v['Ws'][W_name] = np.zeros_like(layer_grad['dWs'][W_name])
                for bias_name in layer_grad['dbs']:
                    v['bs'][bias_name] = np.zeros_like(layer_grad['dbs'][bias_name])

                layer = {}
                layer['layer'] = layer_grad['layer']
                layer['v'] = v
                
        
        for i, layer in enumerate(self.paramaters):
            grads_update = {}

            for W_name in layer['v']['Ws']:
                layer['v']['Ws'][W_name]=self.beta*layer['v']['Ws'][W_name]+(1-self.beta)*grads[i]['dWs'][W_name]
                grads_update[W_name]=self.alpha*layer['v']['Ws'][W_name]
            for b_name in layer['v']['bs']:
                layer['v']['bs'][b_name]=self.beta*layer['v']['bs'][b_name]+(1-self.beta)*grads[i]['dbs'][b_name]
                grads_update[b_name]=self.alpha * layer['v']['bs'][b_name]
            layer.update_params(grads_update)

class RMSProp(Optimizers_):

    def __init__(self, alpha=0.01, beta=0.9, epsilon=1e-9):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.paramaters = []

    def step(self, grads):
        if len(self.paramaters) == 0:
            for layer_grad in grads:
                s = {'Ws':{},'bs':{}}

                for W_name in layer_grad['dWs']:
                    s['Ws'][W_name] = np.zeros_like(layer_grad['dWs'][W_name])
                for bias_name in layer_grad['dbs']:
                    s['bs'][bias_name] = np.zeros_like(layer_grad['dbs'][bias_name])

                layer = {}
                layer['layer'] = layer_grad['layer']
                layer['s'] = s
        
        for i, layer in enumerate(self.paramaters):
            grads_update = {}

            for W_name in layer['s']['Ws']:
                layer['s']['Ws'][W_name]=self.beta*layer['s']['Ws'][W_name]+(1-self.beta)*grads[i]['dWs'][W_name]**2
                grads_update[W_name]=self.alpha*(grads[i]['dWs'][W_name]/(np.sqrt(layer['s']['Ws'][W_name]) + self.epsilon))
            if layer['v']['bs'] != None:
                for b_name in layer['v']['bs']:
                    layer['s']['bs'][b_name]=self.beta*layer['s']['bs'][b_name]+(1-self.beta)*grads[i]['dbs'][b_name]**2
                    grads_update[b_name]=self.alpha * (grads[i]['dbs'][b_name]/(np.sqrt(layer['s']['bs'][b_name]) + self.epsilon))
            layer.update_params(grads_update)

class Adam(Optimizers_):
    
    def __init__(self, alpha=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-9):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 1
        self.parameters = []

    def step(self, grads):

        if len(self.parameters) == 0:
            for layer_grad in grads:
                # create v and s for multi weights in a layer
                # every single layer in layer list can have more than 1 weight and bias
                v = {'Ws':{}, 'bs':{}} 
                s = {'Ws':{}, 'bs':{}}
                for weight_name in layer_grad['dWs']:
                    v['Ws'][weight_name] = np.zeros_like(layer_grad['dWs'][weight_name])
                    s['Ws'][weight_name] = np.zeros_like(layer_grad['dWs'][weight_name])
                if layer_grad['dbs'] is not None:
                    for bias_name in layer_grad['dbs']:
                        v['bs'][bias_name] = np.zeros_like(layer_grad['dbs'][bias_name])
                        s['bs'][bias_name] = np.zeros_like(layer_grad['dbs'][bias_name])
                #Create a new dictionary to hold layer optimize paramaters: (layer__object, v, s)
                layer = {}
                layer['layer'] = layer_grad['layer']
                layer['v'] = v
                layer['s'] = s
                #Optimizer's parameters will hold every learnable layer information describe before
                self.parameters.append(layer)
        

        for i, layer in enumerate(self.parameters):
            grads_update = {}

            for W_name in layer['v']['Ws']:
                layer['v']['Ws'][W_name] = self.beta_1*layer['v']['Ws'][W_name]+(1-self.beta_1)*grads[i]['dWs'][W_name]
                layer['s']['Ws'][W_name] = self.beta_2*layer['s']['Ws'][W_name]+(1-self.beta_2)*grads[i]['dWs'][W_name]**2
                v_correct = layer['v']['Ws'][W_name]/(1-self.beta_1**self.t)
                s_correct = layer['s']['Ws'][W_name]/(1-self.beta_2**self.t)
                grads_update[W_name] = self.alpha*(v_correct/(np.sqrt(s_correct)+self.epsilon))
            
            if layer['v']['bs'] is not None:
                for b_name in layer['v']['bs']:
                    layer['v']['bs'][b_name]=self.beta_1*layer['v']['bs'][b_name]+(1-self.beta_1)*grads[i]['dbs'][b_name]
                    layer['s']['bs'][b_name]=self.beta_2*layer['s']['bs'][b_name]+(1-self.beta_2)*grads[i]['dbs'][b_name]**2
                    v_correct = layer['v']['bs'][b_name]/(1-self.beta_1**self.t)
                    s_correct = layer['s']['bs'][b_name]/(1-self.beta_2**self.t)
                    grads_update[b_name]=self.alpha*(v_correct/(np.sqrt(s_correct)+self.epsilon))
            
            layer['layer'].update_params(grads_update)
        self.t += 1