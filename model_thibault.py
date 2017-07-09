import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

class net(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(net, self).__init__()
        with self.init_scope(): 
            self.pretrained = L.VGG16Layers()
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out
            
            
    def __call__(self, x):
        with chainer.no_backprop_mode():
            h = self.pretrained(x)
            h = h['prob']
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h
