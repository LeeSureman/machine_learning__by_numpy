import numpy as np
import math


def relu(m):
    return m * (m > 0)

def sigmoid(m):
    return 1/(1+np.exp(-m))


class NN():
    def __init__(self, input_x, hidden, output_y):
        self.w1 = np.random.uniform(-math.sqrt(6 / (input_x + hidden)), math.sqrt(6 / (input_x + hidden)),
                                    size=[input_x, hidden])
        self.b1 = np.zeros([hidden])
        self.w2 = np.random.uniform(-math.sqrt(6 / (hidden + output_y)), math.sqrt(6 / (hidden + output_y)),
                                    size=[hidden, output_y])
        self.b2 = np.zeros([output_y])

    def forward(self, x):
        node = {}
        h = sigmoid(np.matmul(x, self.w1) + self.b1)
        predict = np.exp(np.matmul(h, self.w2) + self.b2)
        predict = predict / np.sum(predict, axis=-1, keepdims=True)
        node['h'] = h
        node['p'] = predict
        return node

    def backward(self, x, y):
        derivatives = {}
        node = self.forward(x)
        h1 = node['h']
        p = node['p']
        d_p = 1 / 2 * 2 * (p - y)

        d_z2 = np.zeros(shape=p.shape)
        #for gradients about softmax, don't forget the z^i 's gradient from p^j where j!=i
        for i in range(x.shape[0]):
            for j in range(p.shape[1]):
                d_z2[i, j] += (-np.sum(d_p[i] * p[i] * p[i, j]) + d_p[i, j] * p[i, j])
        d_w2 = np.matmul(np.transpose(h1), d_z2)
        d_b2 = np.sum(d_z2, axis=0)
        d_h1 = np.matmul(d_z2, np.transpose(self.w2))
        #for sigmoid
        d_z1 = d_h1 * h1 * (1 - h1)
        # for relu
        # d_z1 = d_h1*(h1>0)
        d_w1 = np.matmul(np.transpose(x), d_z1)
        d_b1 = np.sum(d_z1, axis=0)
        derivatives['w2'] = d_w2
        derivatives['b2'] = d_b2
        derivatives['w1'] = d_w1
        derivatives['b1'] = d_b1
        derivatives['z2'] = d_z2
        return derivatives

    def fit(self, train_x, train_y, epoch, minibatch, lr):
        for i in range(epoch):
            low = 0
            high = minibatch
            for j in range(train_x.shape[0] // minibatch):
                now_x = train_x[low:high]
                now_y = train_y[low:high]
                derivatives = self.backward(now_x, now_y)
                self.w1 -= lr * derivatives['w1']
                self.b1 -= lr * derivatives['b1']
                self.w2 -= lr * derivatives['w2']
                self.b2 -= lr * derivatives['b2']
                low += minibatch
                high += minibatch
                if high > train_x.shape[0]:
                    high = train_x.shape[0]

            print(i, 'th epoch training loss:', self.l2_loss(train_x, train_y))


    def l2_loss(self, x, y):
        return np.sum(1 / 2 * np.square((self.forward(x)['p'] - y)))


x_1 = np.array([[0, 1, 1, 0, 0,
                 0, 0, 1, 0, 0,
                 0, 0, 1, 0, 0,
                 0, 0, 1, 0, 0,
                 0, 1, 1, 1, 0]], dtype=np.float64)
x_2 = np.array([[1, 1, 1, 1, 0,
                 0, 0, 0, 0, 1,
                 0, 1, 1, 1, 0,
                 1, 0, 0, 0, 0,
                 1, 1, 1, 1, 1]], dtype=np.float64)
x_3 = np.array([[1, 1, 1, 1, 0,
                 0, 0, 0, 0, 1,
                 0, 1, 1, 1, 0,
                 0, 0, 0, 0, 1,
                 1, 1, 1, 1, 0]], dtype=np.float64)
x_4 = np.array([[0, 0, 0, 1, 0,
                 0, 0, 1, 1, 0,
                 0, 1, 0, 1, 0,
                 1, 1, 1, 1, 1,
                 0, 0, 0, 1, 0]], dtype=np.float64)
x_5 = np.array([[1, 1, 1, 1, 1,
                 1, 0, 0, 0, 0,
                 1, 1, 1, 1, 0,
                 0, 0, 0, 0, 1,
                 1, 1, 1, 1, 0]], dtype=np.float64)
x_6 = np.array([[0, 0, 1, 1, 0,
                 0, 0, 1, 1, 0,
                 0, 1, 0, 1, 0,
                 0, 0, 0, 1, 0,
                 0, 1, 1, 1, 0]], dtype=np.float64)
x_7 = np.array([[1, 1, 1, 1, 0,
                 0, 0, 0, 0, 1,
                 0, 1, 1, 1, 0,
                 1, 0, 0, 0, 1,
                 1, 1, 1, 1, 1]], dtype=np.float64)
x_8 = np.array([[1, 1, 1, 1, 0,
                 0, 0, 0, 0, 1,
                 0, 1, 1, 1, 0,
                 1, 0, 0, 0, 1,
                 1, 1, 1, 1, 0]], dtype=np.float64)
x_9 = np.array([[0, 1, 1, 1, 0,
                 0, 1, 0, 0, 0,
                 0, 1, 1, 1, 0,
                 0, 0, 0, 1, 0,
                 0, 1, 1, 1, 0]], dtype=np.float64)
x_10 = np.array([[0, 1, 1, 1, 1,
                  0, 1, 0, 0, 0,
                  0, 1, 1, 1, 0,
                  0, 0, 0, 1, 0,
                  1, 1, 1, 1, 0]], dtype=np.float64)
y_1 = np.array([[1, 0, 0, 0, 0]], dtype=np.float64)
y_2 = np.array([[0, 1, 0, 0, 0]], dtype=np.float64)
y_3 = np.array([[0, 0, 1, 0, 0]], dtype=np.float64)
y_4 = np.array([[0, 0, 0, 1, 0]], dtype=np.float64)
y_5 = np.array([[0, 0, 0, 0, 1]], dtype=np.float64)
y_6 = np.array([[1, 0, 0, 0, 0]], dtype=np.float64)
y_7 = np.array([[0, 1, 0, 0, 0]], dtype=np.float64)
y_8 = np.array([[0, 0, 1, 0, 0]], dtype=np.float64)
y_9 = np.array([[0, 0, 0, 1, 0]], dtype=np.float64)
y_10 = np.array([[0, 0, 0, 0, 1]], dtype=np.float64)
train_x = np.concatenate([x_1, x_2, x_3, x_4, x_5], axis=0)
train_y = np.concatenate([y_1, y_2, y_3, y_4, y_5], axis=0)
test_x = np.concatenate([x_6, x_7, x_8, x_9, x_10], axis=0)
test_y = np.concatenate([y_6, y_7, y_8, y_9, y_10], axis=0)
x_123 = np.concatenate([x_1, x_2, x_3], axis=0)
y_123 = np.concatenate([y_1, y_2, y_3], axis=0)
nn = NN(25, 10, 5)
nn.fit(train_x, train_y, 50, 1, 1)
print(nn.l2_loss(test_x,test_y))
