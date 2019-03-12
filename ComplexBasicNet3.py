"""functions used to construct different architectures
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages


class BasicNet(object):
    weight_decay = 5 * 1e-20
    weight_init = 0.1  # weight init for biasis
    leaky_alpha = 0.1
    eps = 1e-7,
    debugflag = False
    layercount = 0

    def __init__(self):
        self.pretrain_var_collection = []
        self.initial_var_collection = []
        self.trainable_var_collection = []
        self.var_rename = {}
        self.paranumber = 0
        self.ranseed = 0
        self.re = tf.constant([0], dtype=tf.float32)
        self.re2 = tf.constant([0], dtype=tf.float32)
        self.layername = []
    # self.weight_decay = FLAGS.weight_decay
    # self.weight_init = FLAGS.weight_init
    # self.leaky_alpha = FLAGS.leaky_alpha

    def leaky_relu(self, x, alpha, dtype=tf.float32):
        x = tf.cast(x, dtype=dtype)
        bool_mask = (x > 0)
        mask = tf.cast(bool_mask, dtype=dtype)
        return 1.0 * mask * x + alpha * (1 - mask) * x


    def _variable_on_cpu(self, name, shape, initializer, pretrain=False, trainable=True):

        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
            paranum = 1
            for index in shape:
                paranum = paranum * index
            self.paranumber = self.paranumber + paranum
        # self.var_rename['inference/' + var.op.name] = var #for translate
        #print(shape)
        if tf.get_variable_scope().reuse == False:
            if pretrain:
                self.pretrain_var_collection.append(var)
            else:
                self.initial_var_collection.append(var)
            # if trainable:
            #     self.trainable_var_collection.append(var)
        return var

    def _variable_with_weight_decay(self, name, shape, initializer=tf.contrib.layers.xavier_initializer(), pretrain=False,
                                    trainable=True):
        var = self._variable_on_cpu(name, shape, initializer, pretrain, trainable)
        wd = self.weight_decay
        if wd and not tf.get_variable_scope().reuse:
            WeightDecay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            # weight_decay = tf.reduce_mean((var**2)*wd, name='weight_loss')
            WeightDecay.set_shape([])
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, WeightDecay)
        return var

    def conv_layer(self, scope_name, inputs, shape = [3, 3, 100 ,100], strides = [1,1,1,1],
                   initializer=tf.contrib.layers.xavier_initializer(), linear=False, pretrain=False,
                   batchnormalization=False, trainable=True, scale = True):
        with tf.variable_scope(scope_name) as scope:
            input_channels = inputs.get_shape()[3].value
            assert input_channels == shape[2]
            weights = self._variable_with_weight_decay('weights', shape=shape,
                                                       initializer=initializer, pretrain=pretrain, trainable=trainable)
            biases = self._variable_on_cpu('biases', [1,1,1,shape[-1]], tf.constant_initializer(self.weight_init),
                                           pretrain, trainable)
            pad_size = [shape[0] // 2, shape[1] // 2]
            pad_mat = np.array([[0, 0], pad_size, pad_size, [0, 0]])
            inputs_pad = tf.pad(inputs, pad_mat)
            conv = tf.nn.conv2d(inputs_pad, weights, strides = strides, padding='VALID')
            if scale:
                scaleweight = self._variable_on_cpu('scaleweight', [1,1,1,shape[-1]], tf.constant_initializer(1),
                                               pretrain, trainable)
                conv_biased = conv * scaleweight + biases
            else:
                conv_biased = conv + biases

            if batchnormalization:
                conv_biased = tf.layers.batch_normalization(conv_biased, training=trainable)
            if linear:
                output = conv_biased
            else:
                output = self.leaky_relu(conv_biased, self.leaky_alpha)

        if self.debugflag:
            self.layercount = self.layercount + 1
            nant = tf.is_nan(output)
            inft = tf.is_inf(output)
            numnan = tf.expand_dims(tf.reduce_sum(tf.cast(nant, tf.float32)), 0)
            numinf = tf.expand_dims(tf.reduce_sum(tf.cast(inft, tf.float32)), 0)
            self.layername = self.layername + [tf.get_variable_scope().name + '_' + str(self.layercount)]
            self.re = tf.concat([self.re, numnan], axis=0)
            self.re2 = tf.concat([self.re2, numinf], axis=0)
        return output

    def transpose_conv_layer(self, scope_name, inputs, shape=(3, 3, 100, 100), strides=(1, 2, 2, 1), linear=False, pretrain=False,
                             trainable=True):
        # Filter size:A 4-D Tensor with the same type as value and shape [height, width, output_channels, in_channels],different from conv.
        with tf.variable_scope(scope_name) as scope:
            input_channels = inputs.get_shape()[3].value
            assert input_channels == shape[2]
            num_features = shape[3]
            weights = self._variable_with_weight_decay('weights',
                                                       shape=(shape[0], shape[1], shape[3], shape[2]),pretrain=pretrain,
                                                       trainable=trainable)
            biases = self._variable_on_cpu('biases', [num_features], tf.constant_initializer(self.weight_init),
                                           pretrain, trainable)
            # scope.reuse_variables()
            batch_size = tf.shape(inputs)[0]
            inputsshape = inputs.get_shape().as_list()
            output_shape = [inputsshape[0], inputsshape[1] * strides[1], inputsshape[2] * strides[2], num_features]
            conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=strides, padding='SAME')
            conv_biased = tf.nn.bias_add(conv, biases, name='linearout')
            if linear:
                output = conv_biased
            else:
                output = self.leaky_relu(conv_biased, self.leaky_alpha)

        if self.debugflag:
            self.layercount = self.layercount + 1
            nant = tf.is_nan(output)
            inft = tf.is_inf(output)
            numnan = tf.expand_dims(tf.reduce_sum(tf.cast(nant, tf.float32)), 0)
            numinf = tf.expand_dims(tf.reduce_sum(tf.cast(inft, tf.float32)), 0)
            self.layername = self.layername + [tf.get_variable_scope().name + '_' + str(self.layercount)]
            self.re = tf.concat([self.re, numnan], axis=0)
            self.re2 = tf.concat([self.re2, numinf], axis=0)
        return output


    def convgroup2_layer(self, scope_name, inputs, shape = [3, 3, 100 ,100], strides = [1,1,1,1],       ###group conv
                   initializer=tf.contrib.layers.xavier_initializer(), linear=False, pretrain=False,
                   batchnormalization=False, trainable=True):
        with tf.variable_scope(scope_name) as scope:
            input_channels = inputs.get_shape()[3].value
            input_dim = input_channels // 2
            assert input_dim == shape[2]
            input_gp1 = inputs[..., :input_dim]
            input_gp2 = inputs[..., input_dim:]
            W_gp1 = self._variable_with_weight_decay('weights_gp1', shape=shape, initializer=initializer, pretrain=pretrain, trainable=trainable)
            W_gp2 = self._variable_with_weight_decay('weights_gp2', shape=shape, initializer=initializer, pretrain=pretrain, trainable=trainable)
            biases_gp1 = self._variable_on_cpu('biases_gp1', [shape[-1]], tf.constant_initializer(self.weight_init), pretrain, trainable)
            biases_gp2 = self._variable_on_cpu('biases_gp2', [shape[-1]], tf.constant_initializer(self.weight_init),
                                               pretrain, trainable)
            pad_size = [shape[0] // 2, shape[1] // 2]
            pad_mat = np.array([[0, 0], pad_size, pad_size, [0, 0]])
            inputs_pad_gp1 = tf.pad(input_gp1, pad_mat)
            inputs_pad_gp2 = tf.pad(input_gp2, pad_mat)
            conv1 = tf.nn.conv2d(inputs_pad_gp1, W_gp1, strides = strides, padding='VALID')
            conv2 = tf.nn.conv2d(inputs_pad_gp2, W_gp2, strides = strides, padding='VALID')

            conv_biased = tf.concat([tf.nn.bias_add(conv1, biases_gp1), tf.nn.bias_add(conv2, biases_gp2)], axis=-1)

            if batchnormalization:
                conv_biased = tf.layers.batch_normalization(conv_biased, training=trainable)
            if linear:
                return conv_biased
            conv_rect = self.leaky_relu(conv_biased, self.leaky_alpha)
        # scope.reuse_variables()
        return conv_rect

    def _compute_fans(self, shape, data_format='channels_last'):
        if len(shape) < 2:
            raise ValueError("fan in and fan out can not be computed for weight of size ", len(shape))
        elif len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) in {3, 4, 5}:
            # Assuming convolution kernels (1D, 2D or 3D).
            # TH kernel shape: (depth, input_depth, ...)
            # TF kernel shape: (..., input_depth, depth)
            if data_format == 'channels_first':
                receptive_field_size = np.prod(shape[2:])
                fan_in = shape[1] * receptive_field_size
                fan_out = shape[0] * receptive_field_size
            elif data_format == 'channels_last':
                receptive_field_size = np.prod(shape[:-2])
                fan_in = shape[-2] * receptive_field_size
                fan_out = shape[-1] * receptive_field_size
            else:
                raise ValueError('Invalid data_format: ' + data_format)
        else:
            # No specific assumptions.
            fan_in = np.sqrt(np.prod(shape))
            fan_out = np.sqrt(np.prod(shape))
        return fan_in, fan_out

    def _complex_init(self, shape, act='Leaky_sq'):
        fan_in, fan_out = self._compute_fans(shape)
        if act == 'Linear' or 'sigmoid':
            s = 1. / np.sqrt(fan_in + fan_out)
        elif act == 'RealReLu':
            s = np.sqrt(1. / fan_in)
        elif act == 'RealLeaky':
            s = np.sqrt(1. / ((1 + self.leaky_alpha ** 2) * fan_in))
        elif act == 'ReLu':
            s = np.sqrt(2. / 3 * fan_in)
        elif act == 'Leaky':
            s = np.sqrt(2. / ((3 + self.leaky_alpha ** 2) * fan_in))
        elif act == 'Leaky_sq':
            s = np.sqrt(2. / ((1 + 2 * self.leaky_alpha ** 2) * fan_in))
        else:
            raise ValueError('The type of the init criterion error')
        rng = np.random.RandomState(self.ranseed)
        # W_init_real = rng.rayleigh(scale=s, size=shape) * np.cos(rng.uniform(low=-np.pi, high=np.pi, size=shape))
        # W_init_imag = rng.rayleigh(scale=s, size=shape) * np.sin(rng.uniform(low=-np.pi, high=np.pi, size=shape))
        modulus = rng.rayleigh(scale=s, size=shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=shape)
        W_init_real = modulus * np.cos(phase)
        W_init_imag = modulus * np.sin(phase)
        initializer_real = tf.constant_initializer(value=W_init_real)
        initializer_imag = tf.constant_initializer(value=W_init_imag)
        return initializer_real, initializer_imag

    def _alternate_concatdim_re(self, input_list, axis = -1):
        numtensor = len(input_list)
        shape = input_list[0].get_shape().as_list()
        axises = len(shape)
        if axis < 0:
            axis = axis + axises
        mask = tf.range(start=0, limit=numtensor * shape[axis], delta=1, dtype='int32')
        mask = tf.reshape(mask, [-1, numtensor])
        masklist = tf.unstack(mask, axis=-1)
        inputtensor = tf.stack(input_list, axis=-1)
        transposeaxis = [axis] + list(np.arange(0, axis)) + list(np.arange(axis + 1, axises + 1))
        inputtensor = tf.transpose(inputtensor, perm=transposeaxis)
        input_list = tf.unstack(inputtensor, axis=-1)
        cat = tf.dynamic_stitch(masklist, input_list)
        transposeaxis2 = list(np.arange(0, axis) + 1) + [0] + list(np.arange(axis + 1, axises))
        cat = tf.transpose(cat, perm=transposeaxis2)
        return cat

    def _alternate_concatdim(self, input_list, axis=-1):
        numtensor = len(input_list)
        shape = input_list[0].get_shape().as_list()
        axises = len(shape)
        shape[axis] = numtensor * shape[axis]
        if axis >= 0:
            axis = axis + 1
            if axis == axises:
                axis = -1
        cat = tf.reshape(tf.stack(input_list, axis=axis), shape)
        return cat


    def _normlized(self, mat):  # tensor [batch_size, image_height, image_width, channels] normalize each fea map
        mat_shape = mat.get_shape().as_list()
        if len(mat_shape) == 2:
            tempsum = tf.reduce_sum(mat, axis=0)
            tempsum = tf.reduce_sum(tempsum, axis=0) + self.eps
        elif len(mat_shape) == 3:
            tempsum = tf.reduce_sum(mat, axis=1)
            tempsum = tf.reduce_sum(tempsum, axis=1) + self.eps
            tempsum = tf.reshape(tempsum, [-1, 1, 1])
        elif len(mat_shape) == 4:
            tempsum = tf.reduce_sum(mat, axis=1)
            tempsum = tf.reduce_sum(tempsum, axis=1) + self.eps
            tempsum = tf.reshape(tempsum, [-1, 1, 1, mat_shape[3]])
        return mat / tempsum

    def _normlized_0to1(self, mat):  # tensor [batch_size, image_height, image_width, channels] normalize each fea map
        mat_shape = mat.get_shape().as_list()
        if len(mat_shape)==2:
            tempmin = tf.reduce_min(mat, axis=0)
            tempmin = tf.reduce_min(tempmin, axis=0)
            tempmat = mat - tempmin
            tempmax = tf.reduce_max(tempmat, axis=0)
            tempmax = tf.reduce_max(tempmax, axis=0) + self.eps
        elif len(mat_shape)==3:
            tempmin = tf.reduce_min(mat, axis=1)
            tempmin = tf.reduce_min(tempmin, axis=1)
            tempmin = tf.reshape(tempmin, [-1, 1, 1])
            tempmat = mat - tempmin
            tempmax = tf.reduce_max(tempmat, axis=1)
            tempmax = tf.reduce_max(tempmax, axis=1) + self.eps
            tempmax = tf.reshape(tempmax, [-1, 1, 1])
        elif len(mat_shape)==4:
            tempmin = tf.reduce_min(mat, axis=1)
            tempmin = tf.reduce_min(tempmin, axis=1)
            tempmin = tf.reshape(tempmin, [-1, 1, 1, mat_shape[3]])
            tempmat = mat - tempmin
            tempmax = tf.reduce_max(tempmat, axis=1)
            tempmax = tf.reduce_max(tempmax, axis=1) + self.eps
            tempmax = tf.reshape(tempmax, [-1, 1, 1, mat_shape[3]])
        return tempmat / tempmax

    def ComplexConv2dLayer(self,
                           name,
                           input,
                           shape=(3, 3, 100, 100),  # shape of single part [height width inputchannels/2 outputchannels/2]
                           strides=(1, 1, 1, 1),  # [batch height width inputchannels]
                           use_bias=True,
                           pretrain=False,
                           trainable=True,
                           padding='SAME',
                           act_alllayer='Leaky_sq'
                           ):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape().as_list()
            input_dim = int(input_shape[-1]) // 2
            assert input_dim == shape[2]
            initializer_real, initializer_imag = self._complex_init(shape, act=act_alllayer)
            W_r = self._variable_with_weight_decay('weights_real', shape=shape, initializer=initializer_real,
                                                   pretrain=pretrain, trainable=trainable)
            W_i = self._variable_with_weight_decay('weights_imag', shape=shape, initializer=initializer_imag,
                                                   pretrain=pretrain, trainable=trainable)
            cat_W = tf.concat([tf.concat([W_r, -W_i], axis=-2), tf.concat([W_i, W_r], axis=-2)], axis=-1)
            output = tf.nn.conv2d(input, cat_W, strides=strides, padding=padding)
            if use_bias:
                b = self._variable_on_cpu('biases_real', [shape[-1]*2], tf.constant_initializer(self.weight_init),
                                            pretrain, trainable)
                output = tf.nn.bias_add(output, b)
            if self.debugflag:
                self.layercount = self.layercount + 1
                nant = tf.is_nan(output)
                inft = tf.is_inf(output)
                numnan = tf.expand_dims(tf.reduce_sum(tf.cast(nant, tf.float32)),0)
                numinf = tf.expand_dims(tf.reduce_sum(tf.cast(inft, tf.float32)),0)
                self.layername = self.layername + [tf.get_variable_scope().name + '_'+ str(self.layercount)]
                self.re = tf.concat([self.re, numnan], axis=0)
                self.re2 = tf.concat([self.re2, numinf], axis=0)

            return output

    def ComplexScaledLayer(self,
                           name,
                           input,# [batch height width inputchannels]
                           pretrain=False,
                           trainable=True
                           ):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape().as_list()
            input_dim = int(input_shape[-1]) // 2
            beta = self._variable_with_weight_decay('scaleBeta', shape=(1,1,1,input_dim), initializer=tf.zeros_initializer,
                                                   pretrain=pretrain, trainable=trainable)
            gamma = self._variable_with_weight_decay('scaleGamma', shape=(1,1,1,input_dim), initializer=tf.ones_initializer,
                                                   pretrain=pretrain, trainable=trainable)
            cat_beta = tf.concat([beta, beta], axis=-1)
            cat_gamma = tf.concat([gamma, gamma], axis=-1)
            output = input*cat_gamma + cat_beta
            if self.debugflag:
                self.layercount = self.layercount + 1
                nant = tf.is_nan(output)
                inft = tf.is_inf(output)
                numnan = tf.expand_dims(tf.reduce_sum(tf.cast(nant, tf.float32)), 0)
                numinf = tf.expand_dims(tf.reduce_sum(tf.cast(inft, tf.float32)), 0)
                self.layername = self.layername + [tf.get_variable_scope().name + '_' + str(self.layercount)]
                self.re = tf.concat([self.re, numnan], axis=0)
                self.re2 = tf.concat([self.re2, numinf], axis=0)
            return output



    def ComplexConv3dLayer(self,
                           name,
                           input,  # [batch, in_depth, in_height, in_width, in_channels]
                           shape=(3, 3, 3, 100, 100), # shape of single part [filter_depth, filter_height, filter_width, in_channels/2, out_channels/2]
                           strides=(1, 1, 1, 1, 10),  # [batch depth height width inputchannels]
                           use_bias=True,
                           pretrain=False,
                           trainable=True,
                           padding='SAME',
                           act_alllayer='Leaky_sq'
                           ):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape().as_list()
            input_dim = int(input_shape[-1]) // 2
            assert input_dim == shape[-2]
            initializer_real, initializer_imag = self._complex_init(shape, act=act_alllayer)
            W_r = self._variable_with_weight_decay('weights_real', shape=shape, initializer=initializer_real,
                                                   pretrain=pretrain, trainable=trainable)
            W_i = self._variable_with_weight_decay('weights_imag', shape=shape, initializer=initializer_imag,
                                                   pretrain=pretrain, trainable=trainable)

            cat_W = tf.concat([tf.concat([W_r, -W_i], axis=-2), tf.concat([W_i, W_r], axis=-2)], axis=-1)
            output = tf.nn.conv3d(input, cat_W, strides=strides, padding=padding)

            if use_bias:
                b = self._variable_on_cpu('biases_real', [shape[-1]*2], tf.constant_initializer(self.weight_init),
                                            pretrain, trainable)
                output = tf.nn.bias_add(output, b)
            return output


    def ComplexDeConv2dLayer(self,
                             name,
                             input,
                             shape=(3, 3, 100, 100),
                             # shape of single part [height width inputchannels/2 outputchannels/2]
                             strides=(1, 2, 2, 1),  # [batch height width inputchannels]
                             use_bias=True,
                             pretrain=False,
                             trainable=True,
                             padding='SAME',
                             act_alllayer='Leaky_sq'
                             ):

        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape().as_list()
            input_dim = int(input_shape[-1]) // 2
            assert input_dim == shape[2]
            initializer_real, initializer_imag = self._complex_init(shape, act=act_alllayer)
            w_shape = (shape[0], shape[1], shape[3], shape[2])
            W_r = self._variable_with_weight_decay('weights_real', shape=w_shape, initializer=initializer_real,
                                                   pretrain=pretrain, trainable=trainable)
            W_i = self._variable_with_weight_decay('weights_imag', shape=w_shape, initializer=initializer_imag,
                                                   pretrain=pretrain, trainable=trainable)
            cat_W = tf.concat([tf.concat([W_r, -W_i], axis=-1), tf.concat([W_i, W_r], axis=-1)], axis=-2)

            output_shape = (input_shape[0], input_shape[1] * strides[1], input_shape[2] * strides[2], shape[-1]*2)
            output = tf.nn.conv2d_transpose(input, cat_W, output_shape, strides=strides, padding=padding)

            if use_bias:
                b = self._variable_on_cpu('biases_real', [shape[-1] * 2], tf.constant_initializer(self.weight_init),
                                          pretrain, trainable)
                output = tf.nn.bias_add(output, b)
            if self.debugflag:
                self.layercount = self.layercount + 1
                nant = tf.is_nan(output)
                inft = tf.is_inf(output)
                numnan = tf.expand_dims(tf.reduce_sum(tf.cast(nant, tf.float32)), 0)
                numinf = tf.expand_dims(tf.reduce_sum(tf.cast(inft, tf.float32)), 0)
                self.layername = self.layername + [tf.get_variable_scope().name + '_' + str(self.layercount)]
                self.re = tf.concat([self.re, numnan], axis=0)
                self.re2 = tf.concat([self.re2, numinf], axis=0)
            return output

    def ComplexSeparableConv2dLayer(self,
                                    name,
                                    input,
                                    shape=(3, 3, 100, 100),
                                    # shape of single part [height width inputchannels outputchannels/2]
                                    strides=(1, 1, 1, 1),  # [batch height width inputchannels]
                                    depth_multiplier=1,
                                    use_bias=True,
                                    pretrain=False,
                                    trainable=True,
                                    padding='SAME',
                                    act_alllayer='Leaky_sq'
                                    ):

        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape().as_list()
            input_dim = int(input_shape[-1])//2
            assert input_dim == shape[2]
            depthwise_shape = (shape[0], shape[1], shape[2], depth_multiplier)
            pointwise_shape = (1, 1, shape[2]*depth_multiplier, shape[3])
            depthwise_strides = (1, strides[1], strides[2], 1)
            pointwise_strides = (1, 1, 1, strides[3])

            initializer_real, initializer_imag = self._complex_init(depthwise_shape, act=act_alllayer)
            W_dr = self._variable_with_weight_decay('weights_depth_real', shape=depthwise_shape,
                                                    initializer=initializer_real, pretrain=pretrain,
                                                    trainable=trainable)
            W_di = self._variable_with_weight_decay('weights_depth_imag', shape=depthwise_shape,
                                                    initializer=initializer_imag, pretrain=pretrain,
                                                    trainable=trainable)
            depthwise_cat_W = tf.concat([tf.concat([W_dr, -W_di], axis=-2), tf.concat([W_di, W_dr], axis=-2)], axis=-1)
            depthoutput = tf.nn.depthwise_conv2d_native(input, depthwise_cat_W, strides=depthwise_strides, padding=padding)
            depthoutput = depthoutput[..., :input_dim * 2] + depthoutput[..., input_dim * 2:] #ririririri

            initializer_real, initializer_imag = self._complex_init(pointwise_shape, act=act_alllayer)
            W_pr = self._variable_with_weight_decay('weights_point_real', shape=pointwise_shape,
                                                    initializer=initializer_real, pretrain=pretrain,
                                                    trainable=trainable)
            W_pi = self._variable_with_weight_decay('weights_point_imag', shape=pointwise_shape,
                                                    initializer=initializer_imag, pretrain=pretrain,
                                                    trainable=trainable)
            pointwise_cat_W = tf.concat([self._alternate_concatdim([W_pr, -W_pi], axis=-2), self._alternate_concatdim([W_pi, W_pr], axis=-2)], axis=-1)
            output = tf.nn.conv2d(depthoutput, pointwise_cat_W, strides=depthwise_strides, padding=padding)

            if use_bias:
                b = self._variable_on_cpu('biases_real', [shape[-1] * 2], tf.constant_initializer(self.weight_init),
                                          pretrain, trainable)
                output = tf.nn.bias_add(output, b)

            return output


    def BatchNormlizationLayer(self, name, input, BN_type='complex',pretrain=False, trainable=True):
        with tf.variable_scope(name) as scope:
            if BN_type == 'complex':
                output = self._ComplexBatchNormLayer(name + 'BN', input, pretrain=pretrain,
                                                     trainable=trainable)
            elif BN_type == 'real':
                output = tf.layers.batch_normalization(input, training=trainable, trainable=trainable)
            elif BN_type == 'none':
                output = input
            else:
                raise ValueError('The type of the activation error')
            # output = input
            if self.debugflag:
                self.layercount = self.layercount + 1
                nant = tf.is_nan(output)
                inft = tf.is_inf(output)
                numnan = tf.expand_dims(tf.reduce_sum(tf.cast(nant, tf.float32)),0)
                numinf = tf.expand_dims(tf.reduce_sum(tf.cast(inft, tf.float32)),0)
                self.layername = self.layername + [tf.get_variable_scope().name + '_'+ str(self.layercount)]
                self.re = tf.concat([self.re, numnan], axis=0)
                self.re2 = tf.concat([self.re2, numinf], axis=0)
        return output


    def _complex_ReLuact(self, input):
        input_shape = input.get_shape().as_list()
        input_dim = int(input_shape[-1]) // 2
        real = input[..., :input_dim]
        imag = input[..., input_dim:]
        condition = tf.logical_or(tf.greater_equal(real, 0.), tf.greater_equal(imag, 0.))
        return tf.concat([tf.where(condition, real, tf.zeros_like(real)), tf.where(condition, imag, tf.zeros_like(imag))],axis=-1)

    def _complex_ReLuact_leaky(self,input):
        input_shape = input.get_shape().as_list()
        input_dim = int(input_shape[-1]) // 2
        real = input[..., :input_dim]
        imag = input[..., input_dim:]
        alpha = self.leaky_alpha
        condition = tf.logical_or(tf.greater_equal(real, 0.), tf.greater_equal(imag, 0.))
        return tf.concat([tf.where(condition, real, alpha * real), tf.where(condition, imag, alpha * imag)],axis=-1)

    def _complex_ReLuact_leakysq(self,input):
        input_shape = input.get_shape().as_list()
        input_dim = int(input_shape[-1]) // 2
        real = input[..., :input_dim]
        imag = input[..., input_dim:]
        alpha = self.leaky_alpha
        condition_real = tf.greater_equal(real, 0.)
        condition_imag = tf.greater_equal(imag, 0.)
        condition = tf.logical_xor(condition_real, condition_imag)
        real = tf.where(condition, real, alpha * real)
        imag = tf.where(condition, imag, alpha * imag)
        condition = tf.logical_or(condition_real, condition_imag)
        return tf.concat([tf.where(condition, real, alpha * alpha * real), tf.where(condition, imag, alpha * alpha * imag)], axis=-1)

    def _complex_Sigmoid(self, input):
        input_shape = input.get_shape().as_list()
        input_dim = int(input_shape[-1]) // 2
        real = input[..., :input_dim]
        imag = input[..., input_dim:]
        exr = tf.exp( -real )
        denom = 1 + exr**2 + 2 * exr * tf.cos(imag) + self.eps
        outreal = (1 + exr * tf.cos(imag)) / denom
        outimag = exr * tf.sin(imag) / denom
        return tf.concat([outreal, outimag],axis=-1)



    def _complex_bn(self,
                    real_centred,
                    imag_centred,
                    Vrr,
                    Vii,
                    Vri,
                    gamma_rr,
                    gamma_ii,
                    gamma_ri,
                    beta_real,
                    beta_imag,
                    shape
                    ):
        trace = Vrr + Vii
        delta = (Vrr * Vii) - (Vri ** 2)
        s = tf.sqrt(delta)  # Determinant of square root matrix
        t = tf.sqrt(trace + 2 * s)
        inverse_st = 1.0 / (s * t)
        Wrr = (Vii + s) * inverse_st
        Wii = (Vrr + s) * inverse_st
        Wri = -Vri * inverse_st
        broadcast_Wrr = tf.reshape(Wrr, shape)
        broadcast_Wri = tf.reshape(Wri, shape)
        broadcast_Wii = tf.reshape(Wii, shape)
        real_normed = broadcast_Wrr * real_centred + broadcast_Wri * imag_centred
        imag_normed = broadcast_Wri * real_centred + broadcast_Wii * imag_centred
        broadcast_gamma_rr = tf.reshape(gamma_rr, shape)
        broadcast_gamma_ri = tf.reshape(gamma_ri, shape)
        broadcast_gamma_ii = tf.reshape(gamma_ii, shape)
        broadcast_beta_real = tf.reshape(beta_real, shape)
        broadcast_beta_imag = tf.reshape(beta_imag, shape)
        BN_real = broadcast_gamma_rr * real_normed + broadcast_gamma_ri * imag_normed + broadcast_beta_real
        BN_imag = broadcast_gamma_ri * real_normed + broadcast_gamma_ii * imag_normed + broadcast_beta_imag
        return BN_real, BN_imag

    def _ComplexBatchNormLayer(
            self,
            name,
            input,
            decay=0.9,
            pretrain=False,
            trainable=True,
            beta_init=0,
            gamma_diag_init=1 / np.sqrt(2),
            gamma_off_init=0,
            moving_mean_init=0,
            moving_variance_init=1 / np.sqrt(2),
            moving_covariance_init=0
    ):
        input_shape = input.get_shape().as_list()
        input_dim = int(input_shape[-1]) // 2
        input_real = input[..., :input_dim]
        input_imag = input[..., input_dim:]
        params_shape = (input_dim,)
        with tf.variable_scope(name) as scope:
            axis = list(range(len(input_shape) - 1))
            beta_real = self._variable_on_cpu('beta_real', shape=params_shape,
                                              initializer=tf.constant_initializer(beta_init), pretrain=pretrain,
                                              trainable=trainable)
            beta_imag = self._variable_on_cpu('beta_imag', shape=params_shape,
                                              initializer=tf.constant_initializer(beta_init), pretrain=pretrain,
                                              trainable=trainable)
            gamma_rr = self._variable_on_cpu('gamma_rr', shape=params_shape,
                                             initializer=tf.constant_initializer(gamma_diag_init), pretrain=pretrain,
                                             trainable=trainable)
            gamma_ii = self._variable_on_cpu('gamma_ii', shape=params_shape,
                                             initializer=tf.constant_initializer(gamma_diag_init), pretrain=pretrain,
                                             trainable=trainable)
            gamma_ri = self._variable_on_cpu('gamma_ri', shape=params_shape,
                                             initializer=tf.constant_initializer(gamma_off_init), pretrain=pretrain,
                                             trainable=trainable)
            moving_mean_real = self._variable_on_cpu('moving_mean_real', shape=params_shape,
                                                     initializer=tf.constant_initializer(moving_mean_init),
                                                     pretrain=pretrain, trainable=False)
            moving_mean_imag = self._variable_on_cpu('moving_mean_imag', shape=params_shape,
                                                     initializer=tf.constant_initializer(moving_mean_init),
                                                     pretrain=pretrain, trainable=False)
            moving_mean_Vrr = self._variable_on_cpu('moving_mean_Vrr', shape=params_shape,
                                                    initializer=tf.constant_initializer(moving_variance_init),
                                                    pretrain=pretrain, trainable=False)
            moving_mean_Vii = self._variable_on_cpu('moving_mean_Vii', shape=params_shape,
                                                    initializer=tf.constant_initializer(moving_variance_init),
                                                    pretrain=pretrain, trainable=False)
            moving_mean_Vri = self._variable_on_cpu('moving_mean_Vri', shape=params_shape,
                                                    initializer=tf.constant_initializer(moving_covariance_init),
                                                    pretrain=pretrain, trainable=False)

            mu_real = tf.reduce_mean(input_real, axis=axis)
            mu_imag = tf.reduce_mean(input_imag, axis=axis)
            broadcast_mu_shape = [1] * len(input_shape)
            broadcast_mu_shape[-1] = input_dim
            broadcast_mu_real = tf.reshape(mu_real, broadcast_mu_shape)
            broadcast_mu_imag = tf.reshape(mu_imag, broadcast_mu_shape)
            real_centred = input_real - broadcast_mu_real
            imag_centred = input_imag - broadcast_mu_imag
            real_centred_square = real_centred ** 2
            imag_centred_square = imag_centred ** 2

            Vrr = tf.reduce_mean(real_centred_square, axis=axis) + self.eps
            Vii = tf.reduce_mean(imag_centred_square, axis=axis) + self.eps
            Vri = tf.reduce_mean(real_centred * imag_centred, axis=axis)
            update_moving_mean_real = moving_averages.assign_moving_average(
                moving_mean_real, mu_real, decay, zero_debias=False)
            update_moving_mean_imag = moving_averages.assign_moving_average(
                moving_mean_imag, mu_imag, decay, zero_debias=False)
            update_moving_mean_Vrr = moving_averages.assign_moving_average(
                moving_mean_Vrr, Vrr, decay, zero_debias=False)
            update_moving_mean_Vii = moving_averages.assign_moving_average(
                moving_mean_Vii, Vii, decay, zero_debias=False)
            update_moving_mean_Vri = moving_averages.assign_moving_average(
                moving_mean_Vri, Vri, decay, zero_debias=False)

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean_real, update_moving_mean_imag,
                                              update_moving_mean_Vrr, update_moving_mean_Vii, update_moving_mean_Vri]):
                    return tf.identity(mu_real), tf.identity(mu_imag), tf.identity(Vrr), tf.identity(Vii), tf.identity(
                        Vri)

            if trainable:
                mu_real, mu_imag, Vrr, Vii, Vri = mean_var_with_update()
                output_real, output_imag = self._complex_bn(real_centred, imag_centred, Vrr, Vii, Vri,
                                                            gamma_rr, gamma_ii, gamma_ri, beta_real, beta_imag,
                                                            shape=broadcast_mu_shape)
            else:
                real_centred = input_real - tf.reshape(moving_mean_real, broadcast_mu_shape)
                imag_centred = input_imag - tf.reshape(moving_mean_imag, broadcast_mu_shape)
                output_real, output_imag = self._complex_bn(real_centred, imag_centred, moving_mean_Vrr,
                                                            moving_mean_Vii, moving_mean_Vri,
                                                            gamma_rr, gamma_ii, gamma_ri, beta_real, beta_imag,
                                                            shape=broadcast_mu_shape)
            output = tf.concat([output_real, output_imag], axis=-1)
            return output

    def ActivationLayer(self, input, act='Leaky_sq'):
        if act == 'Linear':
            output = input
        elif act == 'ReLu':
            output = self._complex_ReLuact(input)
        elif act == 'Leaky':
            output = self._complex_ReLuact_leaky(input)
        elif act == 'Leaky_sq':
            output = self._complex_ReLuact_leakysq(input)
        elif act == 'RealLeaky':
            output = self.leaky_relu(input,self.leaky_alpha )
        elif act == 'RealReLu':
            output = tf.nn.relu(input)
        elif act == 'sigmoid':
            output = self._complex_Sigmoid(input)
        else:
            raise ValueError('The type of the activation error')
        if self.debugflag:
            self.layercount = self.layercount + 1
            nant = tf.is_nan(output)
            inft = tf.is_inf(output)
            numnan = tf.expand_dims(tf.reduce_sum(tf.cast(nant, tf.float32)), 0)
            numinf = tf.expand_dims(tf.reduce_sum(tf.cast(inft, tf.float32)), 0)
            self.layername = self.layername + [tf.get_variable_scope().name + '_' + str(self.layercount)]
            self.re = tf.concat([self.re, numnan], axis=0)
            self.re2 = tf.concat([self.re2, numinf], axis=0)
        return output


    def _argmaxpool(self, inputs, ksize, strides, padding='SAME'):
        with tf.name_scope('argmax_pool'):
            inputdim = len(inputs.get_shape().as_list())
            assert inputdim == len(ksize) and inputdim == len(strides)
            if inputdim == 4:
                downsample = tf.nn.max_pool(inputs, ksize, strides, padding)
                out_shape = downsample.get_shape().as_list()
                grad_sum = tf.reduce_sum(downsample)
                mask = tf.gradients(grad_sum, inputs, colocate_gradients_with_ops=True)
                indices = tf.where(tf.greater(mask[0], 0))
            elif inputdim == 5:
                downsample = tf.nn.max_pool3d(inputs, ksize, strides, padding)
                out_shape = downsample.get_shape().as_list()
                grad_sum = tf.reduce_sum(downsample)
                mask = tf.gradients(grad_sum, inputs, colocate_gradients_with_ops=True)
                indices = tf.where(tf.greater(mask[0], 0))
            else:
                raise ValueError('The inputdim should be 4 or 5')
            return out_shape, indices

    def ComplexPoolLayer(self,
                         name,
                         input,
                         ksize,  # [batch, depth, height, width, channels]
                         strides,
                         padding='SAME',
                         pool='ArgMaxAbs'
                         ):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape().as_list()
            input_dim = int(input_shape[-1]) // 2
            input_real = input[..., :input_dim]
            input_imag = input[..., input_dim:]

            if pool == 'ArgMaxAbs':  # argmax(|z|)
                input_complex = tf.complex(input_real, input_imag)
                shape, indices = self._argmaxpool(tf.abs(input_complex), ksize=ksize, strides=strides, padding=padding)

                output_real = tf.reshape(tf.gather_nd(input_real, indices), shape)
                output_imag = tf.reshape(tf.gather_nd(input_imag, indices), shape)
            elif pool == 'split_ArgMaxAbs':  # 'split_ArgMax', argmax(|Re(z)|+|Im(z)|)
                shape, indices = self._argmaxpool(tf.add(tf.abs(input_real), tf.abs(input_imag)), ksize=ksize,
                                                  strides=strides,
                                                  padding=padding)
                output_real = tf.reshape(tf.gather_nd(input_real, indices), shape)
                output_imag = tf.reshape(tf.gather_nd(input_imag, indices), shape)
            elif pool == 'Mean':
                output_real = tf.nn.avg_pool(input_real, ksize=ksize, strides=strides, padding=padding)
                output_imag = tf.nn.avg_pool(input_imag, ksize=ksize, strides=strides, padding=padding)
            else:
                raise ValueError(
                    'The type of the Complex2DPoolLayer should be either `ArgMaxAbs`, `split_ArgMaxAbs` or `Mean`'
                )
            output = tf.concat([output_real, output_imag], axis=-1)
            #
            # if self.debugflag:
            #     a = tf.reshape(output, shape=[1, -1])
            #     self.layername = self.layername + [tf.get_variable_scope().name]
            #     self.re = tf.concat([self.re, a[..., 0]], axis=0)
        return output


    def ComplexProjectLayer(
        self,
        name,
        inputs,
        pretrain=False,
        trainable=True,
        use_bias=True,
        type='Abs'
      ):
        input_shape = inputs.get_shape().as_list()
        input_dim = int(input_shape[-1]) // 2
        input_real = inputs[..., :input_dim]
        input_imag = inputs[..., input_dim:]
        with tf.variable_scope(name) as vs:
            if type == 'Abs':
                output = tf.square(tf.abs(tf.complex(input_real, input_imag)))
            elif type == 'Fc':
                weights = self._variable_with_weight_decay('FC',
                                                          shape=[1, 1, 1, 2, 1],
                                                          initializer= tf.constant_initializer(1 / 2), pretrain=pretrain,
                                                          trainable=trainable)
                input_complex = tf.stack([input_real, input_imag], -1)
                input_complex = tf.transpose(input_complex,[0,3,1,2,4])
                output = tf.nn.conv3d(input_complex, weights, strides=[1, 1, 1, 1, 1], padding='SAME')
                if use_bias:
                    b = self._variable_on_cpu('biases_real', [1], tf.constant_initializer(self.weight_init),
                                              pretrain, trainable)
                    output = tf.nn.bias_add(output, b)
                output = output[...,0]
                output = tf.transpose(output,[0,2,3,1])
            if type == 'channalFC':
                FcReal = self._variable_with_weight_decay('FcReal', shape=(1, 1, 1, input_dim),
                                                        initializer=tf.constant_initializer(1 / 2),
                                                        pretrain=pretrain, trainable=trainable)
                FcImag = self._variable_with_weight_decay('FcImag', shape=(1, 1, 1, input_dim),
                                                        initializer=tf.constant_initializer(1 / 2),
                                                        pretrain=pretrain, trainable=trainable)
                output = FcReal * input_real + input_imag * FcImag

            if type == 'Bino':
                Bino1 = self._variable_with_weight_decay('Bino1', shape=(1, 1, 1, 1),
                                                          initializer=tf.constant_initializer(1),
                                                          pretrain=pretrain, trainable=trainable)
                Bino2 = self._variable_with_weight_decay('Bino2', shape=(1, 1, 1, 1),
                                                         initializer=tf.constant_initializer(1),
                                                         pretrain=pretrain, trainable=trainable)
                Bino3 = self._variable_with_weight_decay('Bino3', shape=(1, 1, 1, 1),
                                                         initializer=tf.constant_initializer(0),
                                                         pretrain=pretrain, trainable=trainable)
                Bino4 = self._variable_with_weight_decay('Bino4', shape=(1, 1, 1, 1),
                                                         initializer=tf.constant_initializer(0),
                                                         pretrain=pretrain, trainable=trainable)
                Bino5 = self._variable_with_weight_decay('Bino5', shape=(1, 1, 1, 1),
                                                         initializer=tf.constant_initializer(0),
                                                         pretrain=pretrain, trainable=trainable)
                if use_bias:
                    Bino6 = self._variable_with_weight_decay('Bino6', shape=(1, 1, 1, 1),
                                                             initializer=tf.constant_initializer(0),
                                                             pretrain=pretrain, trainable=trainable)
                    output =  Bino1*(input_real**2) + Bino2*(input_imag**2) + Bino3*input_real*input_imag + Bino4*input_real + Bino5*input_imag + Bino6
                else:
                    output =  Bino1*(input_real**2) + Bino2*(input_imag**2) + Bino3*input_real*input_imag + Bino4*input_real + Bino5*input_imag

            if type == 'channelBino':
                Bino1 = self._variable_with_weight_decay('Bino1', shape=(1, 1, 1, input_dim),
                                                          initializer=tf.constant_initializer(1),
                                                          pretrain=pretrain, trainable=trainable)
                Bino2 = self._variable_with_weight_decay('Bino2', shape=(1, 1, 1, input_dim),
                                                         initializer=tf.constant_initializer(1),
                                                         pretrain=pretrain, trainable=trainable)
                Bino3 = self._variable_with_weight_decay('Bino3', shape=(1, 1, 1, input_dim),
                                                         initializer=tf.constant_initializer(0),
                                                         pretrain=pretrain, trainable=trainable)
                Bino4 = self._variable_with_weight_decay('Bino4', shape=(1, 1, 1, input_dim),
                                                         initializer=tf.constant_initializer(0),
                                                         pretrain=pretrain, trainable=trainable)
                Bino5 = self._variable_with_weight_decay('Bino5', shape=(1, 1, 1, input_dim),
                                                         initializer=tf.constant_initializer(0),
                                                         pretrain=pretrain, trainable=trainable)
                if use_bias:
                    Bino6 = self._variable_with_weight_decay('Bino6', shape=(1, 1, 1, input_dim),
                                                             initializer=tf.constant_initializer(0),
                                                             pretrain=pretrain, trainable=trainable)
                    output =  Bino1*(input_real**2) + Bino2*(input_imag**2) + Bino3*input_real*input_imag + Bino4*input_real + Bino5*input_imag + Bino6
                else:
                    output =  Bino1*(input_real**2) + Bino2*(input_imag**2) + Bino3*input_real*input_imag + Bino4*input_real + Bino5*input_imag

            else:
                raise ValueError('The type of the ComplexProjectLayer should be either Abs, channalFC, Fc, Bino or channelBino ')

            if self.debugflag:
                self.layercount = self.layercount + 1
                nant = tf.is_nan(output)
                inft = tf.is_inf(output)
                numnan = tf.expand_dims(tf.reduce_sum(tf.cast(nant, tf.float32)),0)
                numinf = tf.expand_dims(tf.reduce_sum(tf.cast(inft, tf.float32)),0)
                self.layername = self.layername + [tf.get_variable_scope().name + '_'+ str(self.layercount)]
                self.re = tf.concat([self.re, numnan], axis=0)
                self.re2 = tf.concat([self.re2, numinf], axis=0)


        return output


    def _complex_dropout(self, input, keeprate):
        input_shape = input.get_shape().as_list()
        input_dim = int(input_shape[-1]) // 2
        real = input[..., :input_dim]
        imag = input[..., input_dim:]
        dropmask = []
        for i in range(input_shape[0]):
            if i==0:
                singlemask = tf.random_uniform([1, input_shape[1], input_shape[2], input_dim])
                dropmask = singlemask
            else:
                singlemask = tf.random_uniform([1,input_shape[1],input_shape[2],input_dim])
                dropmask = tf.concat([dropmask,singlemask], axis=0)
        outmask = tf.where(tf.less(dropmask,tf.ones_like(dropmask)*keeprate),tf.ones_like(dropmask),tf.zeros_like(dropmask))
        real = real * outmask * (1/keeprate)
        imag = imag * outmask * (1/keeprate)
        return tf.concat([real, imag], axis=-1)

    def _concat_fea(self, fealist):
        reallist = []
        imaglist = []
        for fea in fealist:
            input_shape = fea.get_shape().as_list()
            input_dim = int(input_shape[-1]) // 2
            real = fea[..., :input_dim]
            imag = fea[..., input_dim:]
            reallist = reallist + [real]
            imaglist = imaglist + [imag]
        output = tf.concat(reallist + imaglist, axis=-1)
        return output

    def _split_fea(self, fea, num):
        input_shape = fea.get_shape().as_list()
        input_dim = int(input_shape[-1]) // 2
        real = fea[..., :input_dim]
        imag = fea[..., input_dim:]
        reallist = tf.split(real, num, axis=3)
        imaglist = tf.split(imag, num, axis=3)
        outlist = []
        for i in range(num):
            output = tf.concat([reallist[i], imaglist[i]], axis=-1)
            outlist = outlist + [output]
        return outlist

    def _complex_mult(self, input1, input2):
        input_shape = input1.get_shape().as_list()
        assert input_shape == input2.get_shape().as_list()
        input_dim = int(input_shape[-1]) // 2
        real1 = input1[..., :input_dim]
        imag1 = input1[..., input_dim:]
        real2 = input2[..., :input_dim]
        imag2 = input2[..., input_dim:]
        outreal = real1 * real2 - imag1 * imag2
        outimag = real1 * imag2 + imag1 * real2
        return tf.concat([outreal, outimag], axis=-1)

