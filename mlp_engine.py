import numpy as np

class MLP:
    def __init__(self, layers=[2, 2, 1], activation='sigmoid', loss='mse', optimizer='sgd'):
        """
        layers: 각 층의 노드 수를 담은 리스트 (예: [2, 3, 1])
        activation: 'sigmoid', 'relu', 'tanh', 'softmax' 중 선택
        loss: 'mse', 'cross_entropy' 중 선택
        optimizer: 'sgd', 'momentum', 'adam' 중 선택
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.activation_type = activation
        self.loss_type = loss
        self.optimizer = optimizer
        
        # 활성화 함수 및 미분 함수 정의
        self._setup_activations()
        
        # Weights and Biases initialization (Glorot/Xavier initialization)
        np.random.seed(42)
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            limit = np.sqrt(6 / (layers[i] + layers[i+1]))
            w = np.random.uniform(-limit, limit, (layers[i+1], layers[i]))
            b = np.zeros((layers[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
            
        self._init_optimizer_cache()
        
        # To store intermediate values for visualization
        self.cache = {}

    def _init_optimizer_cache(self):
        self.v_dW = [np.zeros_like(w) for w in self.weights]
        self.v_db = [np.zeros_like(b) for b in self.biases]
        self.m_dW = [np.zeros_like(w) for w in self.weights]
        self.m_db = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def _setup_activations(self):
        # Check if the requested activation is supported
        supported = ['sigmoid', 'relu', 'tanh', 'softmax']
        if self.activation_type not in supported:
            raise ValueError(f"Unsupported activation: {self.activation_type}")
        self.activations = self.activation_map

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def softmax_derivative(self, x):
        return np.ones_like(x)

    @property
    def activation_map(self):
        """Returns the mapping of activation type to (func, deriv_func)."""
        mapping = {
            'sigmoid': (self.sigmoid, self.sigmoid_derivative),
            'relu': (self.relu, self.relu_derivative),
            'tanh': (self.tanh, self.tanh_derivative),
            'softmax': (self.softmax, self.softmax_derivative)
        }
        return mapping

    def forward(self, X):
        """
        X: input vector or matrix (input_size, batch_size)
        """
        self.cache['activations'] = [X]
        self.cache['zs'] = []
        
        act_func = self.activations[self.activation_type][0]
        current_a = X
        
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], current_a) + self.biases[i]
            current_a = act_func(z)
            self.cache['zs'].append(z)
            self.cache['activations'].append(current_a)
            
        return current_a

    def backward(self, y_true):
        """
        y_true: target value (output_size, batch_size)
        """
        activations = self.cache['activations']
        zs = self.cache['zs']
        m = activations[0].shape[1]

        self.cache['gradients'] = {'dW': [], 'db': []}
        self.cache['deltas'] = []
        
        a_last = activations[-1]
        act_deriv_func = self.activations[self.activation_type][1]
        
        # 1. Calculate Output Layer Error (delta^L)
        if self.loss_type == 'mse':
            loss_grad = (a_last - y_true) / m
            delta = loss_grad * act_deriv_func(zs[-1])
        elif self.loss_type == 'cross_entropy':
            # For Cross-Entropy with Sigmoid/Softmax, dL/dz is exactly (a - y) / m
            # We bypass multiplying by act_deriv_func
            delta = (a_last - y_true) / m
        else:
            raise ValueError(f"Unsupported loss: {self.loss_type}")
        
        # Iterate backwards through layers
        for i in reversed(range(self.num_layers - 1)):
            self.cache['deltas'].insert(0, delta)
            
            dW = np.dot(delta, activations[i].T)
            db = np.sum(delta, axis=1, keepdims=True)
            
            self.cache['gradients']['dW'].insert(0, dW)
            self.cache['gradients']['db'].insert(0, db)
            
            if i > 0:
                # delta^{l} = (W^{l+1}.T @ delta^{l+1}) * sigma'(z^l)
                delta = np.dot(self.weights[i].T, delta) * act_deriv_func(zs[i-1])
        
        return self.cache['gradients']

    def update(self, learning_rate=0.1):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        self.t += 1
        
        for i in range(len(self.weights)):
            dW = self.cache['gradients']['dW'][i]
            db = self.cache['gradients']['db'][i]
            
            if self.optimizer == 'sgd':
                self.weights[i] -= learning_rate * dW
                self.biases[i] -= learning_rate * db
                
            elif self.optimizer == 'momentum':
                self.v_dW[i] = beta1 * self.v_dW[i] + (1 - beta1) * dW
                self.v_db[i] = beta1 * self.v_db[i] + (1 - beta1) * db
                self.weights[i] -= learning_rate * self.v_dW[i]
                self.biases[i] -= learning_rate * self.v_db[i]
                
            elif self.optimizer == 'adam':
                self.m_dW[i] = beta1 * self.m_dW[i] + (1 - beta1) * dW
                self.m_db[i] = beta1 * self.m_db[i] + (1 - beta1) * db
                
                self.v_dW[i] = beta2 * self.v_dW[i] + (1 - beta2) * (dW ** 2)
                self.v_db[i] = beta2 * self.v_db[i] + (1 - beta2) * (db ** 2)
                
                m_dW_corr = self.m_dW[i] / (1 - beta1 ** self.t)
                m_db_corr = self.m_db[i] / (1 - beta1 ** self.t)
                v_dW_corr = self.v_dW[i] / (1 - beta2 ** self.t)
                v_db_corr = self.v_db[i] / (1 - beta2 ** self.t)
                
                self.weights[i] -= learning_rate * m_dW_corr / (np.sqrt(v_dW_corr) + epsilon)
                self.biases[i] -= learning_rate * m_db_corr / (np.sqrt(v_db_corr) + epsilon)

    def set_params(self, weights_list, biases_list):
        self.weights = [np.array(w) for w in weights_list]
        self.biases = [np.array(b).reshape(-1, 1) for b in biases_list]
        self._init_optimizer_cache()

    def train(self, X, y_true, epochs=100, learning_rate=0.1):
        if X.ndim == 1: X = X.reshape(-1, 1)
        if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)

        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            
            if self.loss_type == 'mse':
                loss = 0.5 * np.mean((y_pred - y_true)**2)
            elif self.loss_type == 'cross_entropy':
                # Avoid log(0) with epsilon
                eps = 1e-15
                loss = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
            
            losses.append(loss)
            self.backward(y_true)
            self.update(learning_rate)
            
        return losses

    # Removed legacy get_cache_for_viz method