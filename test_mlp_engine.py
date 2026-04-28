import pytest
import numpy as np
from mlp_engine import MLP

def test_sigmoid():
    mlp = MLP()
    # Test sigmoid(0) = 0.5
    assert mlp.sigmoid(np.array([[0.0]])) == pytest.approx(0.5)
    # Test large positive and negative values
    assert mlp.sigmoid(np.array([[100.0]])) == pytest.approx(1.0, abs=1e-5)
    assert mlp.sigmoid(np.array([[-100.0]])) == pytest.approx(0.0, abs=1e-5)

def test_sigmoid_derivative():
    mlp = MLP()
    # If input is 0, sigmoid(0)=0.5, derivative should be 0.5 * (1 - 0.5) = 0.25
    assert mlp.sigmoid_derivative(np.array([[0.0]])) == pytest.approx(0.25)

def test_forward_pass():
    layers = [2, 3, 1]
    mlp = MLP(layers=layers)
    input_size = layers[0]
    hidden_size = layers[1]
    output_size = layers[2]
    batch_size = 1
    
    X = np.array([[0.5], [0.8]]) # shape (2, 1)
    output = mlp.forward(X)
    
    # Check output shape
    assert output.shape == (output_size, batch_size)
    # Check cache contents
    assert 'activations' in mlp.cache
    assert 'zs' in mlp.cache
    # Check shapes of cached values (nodes, batch_size)
    assert mlp.cache['zs'][0].shape == (hidden_size, batch_size)
    assert mlp.cache['activations'][1].shape == (hidden_size, batch_size)
    assert mlp.cache['zs'][1].shape == (output_size, batch_size)
    assert mlp.cache['activations'][-1].shape == (output_size, batch_size)

def test_backward_pass():
    layers = [2, 2, 1]
    mlp = MLP(layers=layers)
    input_size = layers[0]
    hidden_size = layers[1]
    output_size = layers[2]
    
    X = np.array([[0.5], [0.3]])
    y_true = np.array([[1.0]])
    
    mlp.forward(X)
    grads = mlp.backward(y_true)
    
    # grads should be {'dW': [...], 'db': [...]}
    assert 'dW' in grads
    assert 'db' in grads
    assert len(grads['dW']) == len(layers) - 1
    assert len(grads['db']) == len(layers) - 1
    
    # Check gradient shapes
    assert grads['dW'][0].shape == mlp.weights[0].shape
    assert grads['db'][0].shape == mlp.biases[0].shape
    assert grads['dW'][1].shape == mlp.weights[1].shape
    assert grads['db'][1].shape == mlp.biases[1].shape

def test_update():
    layers = [2, 2, 1]
    mlp = MLP(layers=layers)
    # Set weights to known values
    w1 = np.ones((2, 2))
    b1 = np.zeros((2, 1))
    w2 = np.ones((1, 2))
    b2 = np.zeros((1, 1))
    mlp.set_params([w1, w2], [b1, b2])
    
    # Manually inject gradients into cache for testing update
    mlp.cache['gradients'] = {
        'dW': [np.ones((2, 2)) * 0.1, np.ones((1, 2)) * 0.1],
        'db': [np.ones((2, 1)) * 0.1, np.ones((1, 1)) * 0.1]
    }
    
    lr = 0.5
    mlp.update(learning_rate=lr)
    
    # New W1 should be 1 - (0.5 * 0.1) = 0.95
    assert np.allclose(mlp.weights[0], 0.95)
    assert np.allclose(mlp.biases[0], -0.05) # 0 - (0.5 * 0.1)
    assert np.allclose(mlp.weights[1], 0.95)
    assert np.allclose(mlp.biases[1], -0.05)

def test_train_convergence():
    # Test if loss decreases over training
    layers = [2, 2, 1]
    mlp = MLP(layers=layers)
    
    X = np.array([[0.5], [0.1]])
    y_true = np.array([[0.9]])
    
    initial_loss = 0.5 * np.mean((mlp.forward(X) - y_true)**2)
    losses = mlp.train(X, y_true, epochs=50, learning_rate=0.5)
    final_loss = losses[-1]
    
    # Loss should decrease
    assert final_loss < initial_loss

def test_softmax():
    mlp = MLP(activation='softmax')
    x = np.array([[1.0], [2.0], [3.0]])
    out = mlp.softmax(x)
    # Exponentials: e^1, e^2, e^3
    # Denominator: e^1 + e^2 + e^3
    assert np.allclose(np.sum(out, axis=0), 1.0)
    assert out.shape == (3, 1)

def test_optimizers():
    # Test momentum
    mlp_momentum = MLP(layers=[2, 2, 1], optimizer='momentum')
    mlp_momentum.cache['gradients'] = {
        'dW': [np.ones((2, 2)) * 0.1, np.ones((1, 2)) * 0.1],
        'db': [np.ones((2, 1)) * 0.1, np.ones((1, 1)) * 0.1]
    }
    mlp_momentum.update(learning_rate=0.1)
    
    # Test adam
    mlp_adam = MLP(layers=[2, 2, 1], optimizer='adam')
    mlp_adam.cache['gradients'] = {
        'dW': [np.ones((2, 2)) * 0.1, np.ones((1, 2)) * 0.1],
        'db': [np.ones((2, 1)) * 0.1, np.ones((1, 1)) * 0.1]
    }
    mlp_adam.update(learning_rate=0.1)
    
    # Just checking they run without errors and update weights
    assert mlp_momentum.v_dW[0].shape == (2, 2)
    assert mlp_adam.m_dW[0].shape == (2, 2)
    assert mlp_adam.v_dW[0].shape == (2, 2)