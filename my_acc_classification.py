__author__ = 'm.bashari'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

class NNModel:
    Ws = [];
    bs = [];
    layers = [];
    def __init__(self, layers):
        self.layers = layers
        init_model(self)

    def train(self, X, y, num_passes=40000, print_loss=False):
        train_model(self, X, y, num_passes, print_loss)

def init_model(model):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    layers = model.layers;
    hidden_layer_num = len(layers) - 1
    Ws = [1] * hidden_layer_num
    bs = [1] * hidden_layer_num
    for i in range(0, hidden_layer_num):
        Ws[i] = np.random.randn(layers[i], layers[i + 1]) / np.sqrt(layers[i])
        bs[i] = np.zeros((1, layers[i + 1]))
    model.Ws = Ws
    model.bs = bs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_data(num):
    np.random.seed(0)
    X, y = datasets.make_moons(num, noise=0.20)
    return X, y

def visualize(X, y, model):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(lambda x:predict(model,x), X, y)
    plt.title("Neural Network")

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

# Helper function to evaluate the output loss on the dataset
def calculate_loss(current_output, expected_output):
    output_shape = expected_output.shape
    num_output = output_shape[0]#training set size
    dimension_output =  output_shape[1]# output dimension
    loss = np.sum(np.sum(
        (-expected_output) * np.log(current_output)
        -(1 - expected_output) * np.log(1 - current_output), axis=0)/ num_output) / dimension_output
    d_loss = ((1-expected_output)/(1-current_output) - (expected_output/current_output))\
             * current_output * (1 - current_output)
    return loss, d_loss

def forward(model, x):
    Ws = model.Ws
    bs = model.bs
    hidden_layer_num = len(Ws)
    a_output = [1] * hidden_layer_num;
    current_input = x

    for i in range(0, hidden_layer_num - 1):
        w_current = Ws[i]
        b_current = bs[i]
        z_current = current_input.dot(w_current) + b_current
        a_current = sigmoid(z_current)
        a_output[i] = a_current
        current_input = a_current

    #output layer
    z_current = current_input.dot(Ws[hidden_layer_num - 1]) + bs[hidden_layer_num - 1]
    a_output[hidden_layer_num - 1] = softmax(z_current)
    return a_output

def softmax(x):
    exp_scores = np.exp(x);
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

def predict(model, x):
    a_output = forward(model, x);
    return np.argmax(a_output[-1], axis=1)

def backward(model, X, expected_output, a_output):
    Ws = model.Ws
    bs = model.bs
    hidden_layer_num = len(Ws)
    ds = [1] * hidden_layer_num
    current_loss, d_current = calculate_loss(a_output[-1], expected_output)
    ds[hidden_layer_num - 1] = d_current
    for l in range(hidden_layer_num - 2, -1, -1):
        w_current = Ws[l + 1]
        a_current = a_output[l]
        d_current = d_current.dot(w_current.T) * a_current * (1-a_current)
        ds[l] = d_current
    #calc dW && db
    dWs = [1] * hidden_layer_num
    dbs = [1] * hidden_layer_num
    a_last = X
    num_output = len(X)
    for l in range(0, hidden_layer_num):
        d_current = ds[l]
        dWs[l] = np.dot(a_last.T, d_current) / num_output
        dbs[l] = np.sum(d_current, axis=0, keepdims=True) / num_output
        a_last = a_output[l]
    return dWs, dbs, current_loss

def update_model_params(model, dWs, dbs):
    Ws = model.Ws
    bs = model.bs
    hidden_layer_num = len(Ws)
    for l in range(0, hidden_layer_num):
        Ws[l] = Ws[l] - Config.epsilon * (dWs[l] + Config.reg_lambda * Ws[l])
        bs[l] = bs[l] - Config.epsilon * dbs[l]
    model.Ws = Ws
    model.bs = bs
    return model

def transform_output_dimension(y):
    num = len(y);
    output = np.zeros((num, 2))
    output[range(num), y] += 1
    return output

# This function learns parameters for the neural network and returns the model.
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def train_model(model, X, y, num_passes=20000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    num_examples = len(X)
    expected_output = transform_output_dimension(y)

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        a_output = forward(model, X)

        # Backpropagation
        dWs, dbs, current_loss = backward(model, X, expected_output, a_output)

        # Update parameters of the model
        update_model_params(model, dWs, dbs)

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, current_loss))
            #exit(0)

    return model


def classify(X, y):
    # clf = linear_model.LogisticRegressionCV()
    # clf.fit(X, y)
    # return clf

    pass

class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength

def main():
    X, y = generate_data(200)
    model = NNModel([2, 3, 2])

    model.train(X, y, print_loss=True)
    visualize(X, y, model)

if __name__ == "__main__":
    main()
