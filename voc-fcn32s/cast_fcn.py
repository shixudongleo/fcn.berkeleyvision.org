#!/usr/bin/env python
import sys
sys.path.append('../')
import caffe

# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('../ilsvrc-nets/VGG_ILSVRC_16_layers_deploy.prototxt',
                '../ilsvrc-nets/VGG_ILSVRC_16_layers.caffemodel',
                caffe.TEST)

# Load the fully convolutional network to transplant the parameters.
net_full_conv = caffe.Net('train.prototxt',
                          # 'snapshot/train_iter_1.caffemodel',
                           caffe.TEST)

params = ['fc6', 'fc7']

# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
for fc in params:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)


params_full_conv = ['fc6', 'fc7']

# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

net_full_conv.save('../ilsvrc-nets/vgg16-fcn.caffemodel')
