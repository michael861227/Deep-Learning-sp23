import numpy as np 
import matplotlib.pyplot as plt

def generate_linear(n = 100):
    """
    Generate data points which are linearly separable
    :param n: number of points
    :return: inputs and labels
    """
    pts = np.random.uniform(0, 1, (n, 2))
    inputs, labels = [], []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        
        if pt[0] > pt[1]:
            labels.append(0)
        else: 
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    """
    Generate data points based on XOR situation
    :param n: number of points
    :return: inputs and labels
    """
    
    inputs, labels = [], []
    
    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)
        
        if 0.1 * i == 0.5:
            continue
            
        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_results(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else: 
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict Result', fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else: 
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.show()


class Layer:
    def __init__(self, input_num, output_num, activation = 'sigmoid', learning_rate = '0.1'):
        self.weight = np.random.normal(0, 1, (input_num + 1, output_num))
        self.activation = activation
        self.learning_rate = learning_rate 
    
    def forward(self, inputs):
        """
        inputs: np.ndarray
        outputs: results compute by this layer
        """
        self.forward_grad = np.append(inputs, np.ones((inputs.shape[0], 1)), axis=1)
        if self.activation == 'sigmoid':
            self.output = self.sigmoid(np.matmul(self.forward_grad, self.weight))
        
        elif self.activation == 'tanh':
            self.output = self.tanh(np.matmul(self.forward_grad, self.weight))
        
        else:
            # Without activation function
            self.output = np.matmul(self.forward_grad, self.weight)

        return self.output
    
    def backward(self, derivative_loss):
        """
        derivative_loss: loss from next layer (np.ndarray)
        return: loss of this layer
        """
        if self.activation == 'sigmoid':
            self.backward_grad = np.multiply(self.derivative_sigmoid(self.output), derivative_loss)
        
        elif self.activation == 'tanh':
            self.backward_grad = np.multiply(self.derivative_tanh(self.output), derivative_loss)
        
        else:
            # Without activation function
            self.backward_grad = derivative_loss

        return np.matmul(self.backward_grad, self.weight[:-1].T)
        
    def update(self):
        """
        Update weight
        """
        gradient = np.matmul(self.forward_grad.T, self.backward_grad)
        self.weight -= self.learning_rate * gradient
    
    def sigmoid(self, x):
        """
        Calculate sigmoid function
        y = 1 / (1 + e^(-x))
        :param x: input data
        :return: sigmoid results
        """
        return 1.0 / (1.0 + np.exp(-x))

    def derivative_sigmoid(self, y):
        """
        Calculate the derivative of sigmoid function
        y' = y(1 - y)
        :param y: value of the sigmoid function
        :return: derivative sigmoid result
        """
        return np.multiply(y, 1.0 - y)

    def tanh(self, x):
        """
        Calculate tanh function
        y = tanh(x)
        :param x: input data
        :return: tanh results
        """
        return np.tanh(x)

    def derivative_tanh(self, y):
        """
        Calculate the derivative of tanh function
        y' = 1 - y^2
        :param y: value of the tanh function
        :return: derivative tanh result
        """
        return 1.0 - y ** 2



class DNN:
    def __init__(self, epoch = 100000, learning_rate = 0.1, input_unit = 2, hidden_unit = 4, activation = 'sigmoid', layer_num = 2):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.activation = activation
        self.learning_epoch, self.learning_loss = [], []
        
        # Input layer
        self.layers = [Layer(input_unit, hidden_unit, activation, learning_rate)]
        
        # Hidden layer
        for _ in range(layer_num - 1):
            self.layers.append(Layer(hidden_unit, hidden_unit, activation, learning_rate))
        
        # Output layer
        self.layers.append(Layer(hidden_unit, 1, 'sigmoid', learning_rate))
    
    def forward(self, inputs):
        """
        Forward feed
        :param inputs: input data
        :return: predict labels
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, derivative_loss):
        """
        Backward propagation
        :param derivative_loss: loss form next layer
        :return: None
        """
        for layer in self.layers[::-1]:
            derivative_loss = layer.backward(derivative_loss)

    def update(self):
        """
        Update all weights in the neural network
        :return: None
        """
        for layer in self.layers:
            layer.update()
    
    def train(self, inputs, labels, threshold = 0.0005):
        for epoch in range(self.epoch):
            predict = self.forward(inputs)
            
            # mse loss
            loss = np.mean((predict - labels) ** 2)
            self.backward(2 * (predict - labels) / len(labels))
            self.update()
    
            if epoch % 500 == 0:
                print(f'Epoch {epoch:<5d}        loss : {loss}')
                self.learning_epoch.append(epoch)
                self.learning_loss.append(loss)
            
            if loss < threshold:
                break
    
    def test(self, inputs, labels):
        prediction = self.forward(inputs)
        loss = np.mean((prediction - labels) ** 2)
        
        for i, (gt, pred) in enumerate(zip(labels, inputs)):
            print(f'Iter{i:<3d}  |  Ground Truth:  {gt[0]}  |   Prediction:  {pred[1]:.3f}')
        print(f'Loss : {loss}   Accuracy : {float(np.sum(np.round(prediction) == labels)) * 100 / len(labels)}%')
        return np.round(prediction)

    def plot_lr_curve(self):
        # Plot learning curve
        plt.figure()
        plt.title('Learning curve', fontsize=18)
        plt.plot(self.learning_epoch, self.learning_loss)

        plt.show()

if __name__ == '__main__':
    epoch = 100000
    learning_rate = 0.1
    hidden_unit = 4
    activation = 'sigmoid'
    
    nn_linear = DNN(epoch = epoch, learning_rate = learning_rate, hidden_unit = hidden_unit,
                    activation = activation)
    
    nn_XOR = DNN(epoch = epoch, learning_rate = learning_rate, hidden_unit = hidden_unit,
                    activation = activation)
    
    x_linear, y_linear = generate_linear()
    x_XOR, y_XOR = generate_XOR_easy()
    
    nn_linear.train(x_linear, y_linear)
    linear_pred = nn_linear.test(x_linear, y_linear)
    nn_linear.plot_lr_curve()
    show_results(x_linear, y_linear, linear_pred)
    

    nn_XOR.train(x_XOR, y_XOR)
    XOR_pred = nn_XOR.test(x_XOR, y_XOR)
    nn_XOR.plot_lr_curve()
    show_results(x_XOR, y_XOR, XOR_pred)

