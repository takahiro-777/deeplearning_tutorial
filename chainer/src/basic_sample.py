# -*- coding:utf-8 -*-

import numpy
from chainer import Variable, optimizers, Chain
import chainer.links as L
import chainer.functions as F

class MLP(Chain):
    def __init__(self, n_in, n_out):
        super(MLP, self).__init__(
              l1=L.Linear(n_in, 10),
              l2=L.Linear(10, n_out),
            )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return h2

#main function
if __name__ == '__main__':
    x = numpy.array([[6, 3, 4], [3, 5, 1]], dtype=numpy.float32)
    x_ = Variable(x)
    print('x_.data=' + str(x_.data))

    d  = numpy.array([[1, 0],[1, 2]], dtype=numpy.float32)
    d_ = Variable(d)
    print('d_.data=' + str(d_.data))

    model = MLP(3, 2)

    #model = L.Linear(3, 2)
    print('model.l1.W.data=' + str(model.l1.W.data))
    print('model.l1.b.data=' + str(model.l1.b.data))

    optimizer = optimizers.SGD()
    optimizer.setup(model)


    for epoch in range(100):
        optimizer.zero_grads()
        u2_ = model(x_)
        z2_ = F.sigmoid(u2_)
        loss = F.mean_squared_error(z2_, d_)
        print("epoch=" + str(epoch) + ", loss=" + str(loss.data))

        loss.backward()
        optimizer.update()

    print('model.l1.W.data=' + str(model.l1.W.data))
    print('model.l1.b.data=' + str(model.l1.b.data))
