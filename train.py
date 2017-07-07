import argparse
import numpy as np

import chainer
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer import training
from chainer.training import extensions
import chainer.links as L

import import_data
import model_thibault

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=25, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=2,
                    help='learning minibatch size')

args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch
print('GPU: {}'.format(args.gpu))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

# Import dataset
X,Y=import_data.import_data()
dataset=zip(X,Y)
train_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)

model = L.Classifier(model_thibault.net(20,4))

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'))

trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/rec_loss', 'main/kl_loss']))

trainer.run()
