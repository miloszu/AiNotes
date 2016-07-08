from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class DeepQModel(Chain):
    def __init__(self, in_size, out_size):
        super(DeepQModel,self).__init__(
            mid=L.Linear(in_size, 64),
            out=L.Linear(64, out_size)
        )

        def reset_state(self):
            self.mid.reset_state

        def __call__(self, x):
            h = F.relu(self.mid(x))
            y = self.out(h)
            return y
