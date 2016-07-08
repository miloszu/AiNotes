import numpy as np
from random import randint
#import matplotlib.pyplot as plt
import pylab as pl
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import minist_data as test_case

class RectifierNetwork(Chain):
    def __init__(self):
        super(RectifierNetwork, self).__init__(
            l1=L.Linear(784, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 10)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y



minist_data = test_case.load_mnist_data()

x_all = minist_data['data'].astype(np.float32) / 255
y_all = minist_data['target'].astype(np.int32)
x_train, x_test = np.split(x_all, [60000])
y_train, y_test = np.split(y_all, [60000])

predictor = RectifierNetwork()
model = L.Classifier(predictor)
optimizer = optimizers.SGD()
optimizer.setup(model)


batch_size = 100
data_size = 60000
pl.ion()
n = 28
img = None

for epoch in range(20):
    print('epoch {}'.format(epoch))
    indexes = np.random.permutation(data_size)
    for i in range(0, data_size, batch_size):
        x = Variable(x_train[indexes[i: i+batch_size]])
        y = Variable(y_train[indexes[i: i + batch_size]])
        optimizer.update(model, x, y)

        if i%20000 == 0:
            test_case = x_test[randint(0, 10000)]
            image = [test_case[k:k + n] for k in range(0, n * n, n)]
            predicted_vector = predictor(np.array([test_case])).data[0]
            predicted_value = np.argmax(predicted_vector)
            pl.title('Prediction: {} (Epoch: {})'.format(predicted_value, epoch))
            if img is None:
                img = pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
            else:
                img.set_data(image)
            pl.pause(.0001)
            pl.draw()

    sum_loss = 0
    sum_accuracy = 0
    for j in range(0, 10000, batch_size):
        x = Variable(x_test[j: j + batch_size ])
        y = Variable(y_test[j: j + batch_size])
        loss = model(x, y)
        sum_loss =+ loss.data * batch_size
        sum_accuracy += model.accuracy.data * batch_size

    mean_loss = sum_loss / 10000
    mean_accuracy = sum_accuracy / 10000
    print('Mean loss: {0}, Mean accuracy: {1}'.format(mean_loss, mean_accuracy))


print("Finished.")



