from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import ComplexBasicNet3 as BasicNet
import math


class Net(BasicNet.BasicNet):  # re
    batch_size = 8
    factor_real = 1
    factor_angle = 0.5 / 10
    factor_mag = 0.1 / 20
    #factor_resmag = 0.3
    # the hyper-parameters in loss function, the factor_real,angle,mag indicates the importance of the loss of projection,phase,amplitude streams. factor_resmag is the importance of the general amplitude (smaller to be more important)

    init_learning_rate = 10 ** (-4)
    pretrain = False
    trainable = True
    use_bias = True  # if use scale layer
    activation = 'RealReLu'  # Linear, Leaky, Leaky_sq, RealLeaky, RealReLu, sigmoid
    poolstyle = 'Mean'  # ArgMaxAbs, split_ArgMaxAbs, Mean
    BN_type = 'real'  # complex, real, none
    prohecttype = 'channelBino'  # Abs, FC,channalFC,Bino, channelBino
    dropout_rate = None  # for drop rate in conv and deconv
    fusion_index = 2  # the number of fusion blocks of decoder
    multi_fusiion = False # if decoder is multi-scale
    predict = []
    loss = []
    complexmap = []
    encoder_compress = 0.5
    growth_rate = 32
    blockfilers = [6, 12, 12]
    def __init__(self):
        super(Net, self).__init__()
        self.global_step = tf.Variable(0, trainable=False)
        self.initial_var_collection.append(self.global_step)
        self.startflagcnn = True
        self.predict = []
        self.loss = []

    def Conv_Bn_Act(self, name, inputs, shape, strides, dropout_rate):
        with tf.variable_scope(name):
            bn1 = self.BatchNormlizationLayer('bn1', inputs, BN_type=self.BN_type, pretrain=self.pretrain,
                                              trainable=self.trainable)
            act1 = self.ActivationLayer(bn1, act=self.activation)
            if self.use_bias:
                scale1 = self.ComplexScaledLayer('scale1', act1, pretrain=self.pretrain, trainable=self.trainable)
            else:
                scale1 = act1
            conv1 = self.ComplexConv2dLayer('conv1', scale1, shape=shape, strides=strides, use_bias=not self.use_bias,
                                            pretrain=self.pretrain, trainable=self.trainable,
                                            act_alllayer=self.activation)
            if dropout_rate:
                conv1 = self._complex_dropout(conv1, 1 - dropout_rate)
        return conv1

    def Deconv_Bn_Act(self, name, inputs, shape, strides, dropout_rate):
        with tf.variable_scope(name):
            bn1 = self.BatchNormlizationLayer('bn1', inputs, BN_type=self.BN_type, pretrain=self.pretrain,
                                              trainable=self.trainable)
            act1 = self.ActivationLayer(bn1, act=self.activation)
            # if self.use_bias:
            #     scale1 = self.ComplexScaledLayer('scale1', act1, pretrain=self.pretrain, trainable=self.trainable)
            # else:
            #     scale1 = act1
            deconv1 = self.ComplexDeConv2dLayer('deconv1', act1, shape=shape, strides=strides,
                                                use_bias=self.use_bias,
                                                pretrain=self.pretrain, trainable=self.trainable,
                                                act_alllayer=self.activation)
            if dropout_rate:
                deconv1 = self._complex_dropout(deconv1, 1 - dropout_rate)
        return deconv1

    def ResBlock(self, name, inputs,
                 feastandard):  # complex residual block, feastandard is a standard filter number for the block
        with tf.variable_scope(name) as scope:
            bn0 = self.BatchNormlizationLayer('bn0', inputs, BN_type=self.BN_type, pretrain=self.pretrain,
                                              trainable=self.trainable)
            act0 = self.ActivationLayer(bn0, act=self.activation)
            input_shape = inputs.get_shape().as_list()
            channalnum = input_shape[-1] // 2
            # assert channalnum == feastandard * 4
            KernelSize = (1, 1) + (channalnum, feastandard)
            comb1 = self.Conv_Bn_Act('comb1', act0, shape=KernelSize, strides=(1, 1, 1, 1),
                                     dropout_rate=self.dropout_rate)
            KernelSize2 = (3, 3) + (feastandard, feastandard)
            comb2 = self.Conv_Bn_Act('comb2', comb1, shape=KernelSize2, strides=(1, 1, 1, 1),
                                     dropout_rate=self.dropout_rate)
            KernelSize3 = (1, 1) + (feastandard, feastandard * 4)
            conv3 = self.ComplexConv2dLayer('conv3', comb2, shape=KernelSize3, use_bias=self.use_bias,
                                            pretrain=self.pretrain,
                                            trainable=self.trainable, act_alllayer=self.activation)
            KernelSize4 = (1, 1) + (channalnum, feastandard * 4)
            conv1x1 = self.ComplexConv2dLayer('conv1x1', inputs, shape=KernelSize4, use_bias=self.use_bias,
                                              pretrain=self.pretrain, trainable=self.trainable,
                                              act_alllayer=self.activation)
            out = tf.add(conv1x1, conv3)
        return out

    def ResBlockDown(self, name, inputs, feastandard):  # complex residual with downsampling
        with tf.variable_scope(name) as scope:
            bn0 = self.BatchNormlizationLayer('bn0', inputs, BN_type=self.BN_type, pretrain=self.pretrain,
                                              trainable=self.trainable)
            act0 = self.ActivationLayer(bn0, act=self.activation)
            input_shape = inputs.get_shape().as_list()
            channalnum = input_shape[-1] // 2
            KernelSize1 = (1, 1) + (channalnum, feastandard)
            poolstrides = (1, 2, 2, 1)
            comb1 = self.Conv_Bn_Act('comb1', act0, shape=KernelSize1, strides=poolstrides,
                                     dropout_rate=self.dropout_rate)
            KernelSize2 = (3, 3) + (feastandard, feastandard)
            comb2 = self.Conv_Bn_Act('comb2', comb1, shape=KernelSize2, strides=(1, 1, 1, 1),
                                     dropout_rate=self.dropout_rate)
            KernelSize3 = (1, 1) + (feastandard, feastandard * 4)
            conv2 = self.ComplexConv2dLayer('conv2', comb2, shape=KernelSize3, use_bias=self.use_bias,
                                            pretrain=self.pretrain,
                                            trainable=self.trainable, act_alllayer=self.activation)
            KernelSize4 = (1, 1) + (channalnum, feastandard * 4)
            conv1x1 = self.ComplexConv2dLayer('conv1x1', inputs, shape=KernelSize4, use_bias=self.use_bias,
                                              strides=poolstrides,
                                              pretrain=self.pretrain, trainable=self.trainable,
                                              act_alllayer=self.activation)
            out = tf.add(conv1x1, conv2)
        return out

    def dense_block(self, name, inputs, nb_layers, nb_filter, growth_rate,
                    dropout_rate):  # nb_filter growth_rate for each complex == real//2,   complex dense block, nb_layers is number of the conv layers, nb_filter is the count of feature channels,
        concat_feat = inputs
        input_shape = inputs.get_shape().as_list()
        channalnum = input_shape[-1] // 2
        assert nb_filter == channalnum
        with tf.variable_scope(name):
            for i in range(nb_layers):
                KernelSize = (1, 1, nb_filter, growth_rate * 4)
                comb1 = self.Conv_Bn_Act('bottleneck' + str(i), concat_feat, shape=KernelSize, strides=(1, 1, 1, 1),
                                         dropout_rate=dropout_rate)
                KernelSize = (3, 3, growth_rate * 4, growth_rate)
                comb2 = self.Conv_Bn_Act('comb' + str(i), comb1, shape=KernelSize, strides=(1, 1, 1, 1),
                                         dropout_rate=dropout_rate)
                concat_feat = self._concat_fea([concat_feat, comb2])
                nb_filter += growth_rate
        return concat_feat, nb_filter

    def transition_block(self, name, inputs, nb_filter, dropout_rate,
                         compression=1.0):  # complex transition block, nb_filter is the count of feature channels
        input_shape = inputs.get_shape().as_list()
        channalnum = input_shape[-1] // 2
        assert nb_filter == channalnum
        nb_filter = int(nb_filter * compression)
        with tf.variable_scope(name):
            KernelSize = (1, 1, channalnum, nb_filter)
            comb1 = self.Conv_Bn_Act('trans', inputs, shape=KernelSize, strides=(1, 1, 1, 1), dropout_rate=dropout_rate)
            pool1 = self.ComplexPoolLayer('pool1', comb1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), pool=self.poolstyle)
        return pool1, nb_filter


    def onstreamDecoder(self, name, features, nb_filter, fuseindex, blocktype = [1,1,1,1], multifusiion = False):  #blocktype: number of blocks and their types 1: decoder block, ２：deconv block
        #print(nb_filter)
        input_shape = features.get_shape().as_list()
        assert nb_filter == input_shape[-1]
        countnb = 0
        multifea = []
        numblock = len(blocktype)
        channalTimes = nb_filter ** (1 / numblock)
        assert channalTimes > 2
        assert fuseindex < numblock
        with tf.variable_scope(name):
            for i in range(numblock):
                bottlechaannal = int(nb_filter / channalTimes * 2 ) + 1
                KernelSize = (1, 1, nb_filter, bottlechaannal)
                features = self.conv_layer('decomb_' + str(i) + '_1', features, shape=KernelSize, strides=(1, 1, 1, 1),
                                           batchnormalization=True, linear=False, scale=True, pretrain=self.pretrain,
                                           trainable=self.trainable)
                nb_filter = bottlechaannal
                if i == numblock - 1:
                    outchannel = 1
                else:
                    outchannel = int(nb_filter / 2) + 1

                if blocktype[i]==1:
                    KernelSize = (3, 3, nb_filter, outchannel)
                    features = self.conv_layer('decomb' + str(i) + '_2', features, shape=KernelSize,
                                               strides=(1, 1, 1, 1),
                                               batchnormalization=True, linear=False, scale=True,
                                               pretrain=self.pretrain,
                                               trainable=self.trainable)
                    nb_filter = outchannel
                elif blocktype[i]==2:
                    KernelSize = (2, 2, nb_filter, outchannel)
                    features = self.transpose_conv_layer('deconv' + str(i), features,
                                                         shape=KernelSize,
                                                         strides=(1, 2, 2, 1),
                                                         pretrain=self.pretrain, trainable=self.trainable)
                    nb_filter = outchannel

                if multifusiion:
                    if i == 0:
                        multifea = features
                        feashape = features.get_shape().as_list()
                    else:
                        tempfea = tf.image.resize_images(features, (feashape[1], feashape[2]))
                        multifea = tf.concat([multifea, tempfea], axis=-1)
                    countnb = countnb + nb_filter
                else:
                    multifea = features
                    countnb = nb_filter

                if i == fuseindex - 1:
                    outfeatures = multifea
        outmap = features
        return outfeatures, outmap


    def encoder_net(self, input):
        input_shapes = input.get_shape().as_list()
        compress = self.encoder_compress
        growth_rate = self.growth_rate
        blockfilers = self.blockfilers
        nb_filter = growth_rate
        with tf.variable_scope('encoder'):
            conv1 = self.ComplexConv2dLayer('conv1', input, shape=(7, 7, input_shapes[-1] // 2, nb_filter),
                                            strides=(1, 2, 2, 1), use_bias=not self.use_bias, pretrain=self.pretrain,
                                            trainable=self.trainable, act_alllayer=self.activation)
            den1, nb_filter = self.dense_block('denseblock1', conv1, blockfilers[0], nb_filter, growth_rate,
                                               self.dropout_rate)
            trans1, nb_filter = self.transition_block('transblock1', den1, nb_filter, self.dropout_rate,
                                                      compression=compress)
            den2, nb_filter = self.dense_block('denseblock2', trans1, blockfilers[1], nb_filter, growth_rate,
                                               self.dropout_rate)
            trans2, nb_filter = self.transition_block('transblock2', den2, nb_filter, self.dropout_rate,
                                                      compression=compress)
            den3, nb_filter = self.dense_block('denseblock3', trans2, blockfilers[2], nb_filter, growth_rate,
                                               self.dropout_rate)
            trans3, nb_filter = self.transition_block('transblock3', den3, nb_filter, self.dropout_rate,
                                                      compression=compress)
        return trans3, nb_filter

    def complexsaliency(self,  phamap, ampmap):
        for i in range(self.batch_size):
            out_angle = phamap[i, :, :, 0]
            out_mag = ampmap[i, :, :, 0]
            outcomplex = tf.complex(out_mag * tf.cos(out_angle), out_mag * tf.sin(out_angle))
            outsalmap = tf.abs(tf.ifft2d(outcomplex))
            outsalmap = tf.expand_dims(outsalmap, -1)
            outsalmap = tf.expand_dims(outsalmap, 0)
            if i == 0:
                compredict = outsalmap
            else:
                compredict = tf.concat([compredict, outsalmap], axis=0)
        return compredict

    def specshift(self,input):
        input_shapes = input.get_shape().as_list()
        assert len(input_shapes) == 2
        h = input_shapes[0]
        w = input_shapes[1]
        p1 = input[0: h // 2, 0: w // 2]
        p2 = input[h // 2: h, 0: w // 2]
        p3 = input[0: h // 2, w // 2: w]
        p4 = input[h // 2: h, w // 2: w]
        p13 = tf.concat([p3, p1], 1)
        p24 = tf.concat([p4, p2], 1)
        pout = tf.concat([p24, p13], 0)
        return pout

    def feaIDFT(self,  feapha, feaamp):
        input_shapes = feapha.get_shape().as_list()
        for i in range(self.batch_size):
            outfeachannel = []
            for j in range(input_shapes[-1]):
                out_angle = feapha[i, :, :, j]
                out_mag = feaamp[i, :, :, j]
                outcomplex = tf.complex(out_mag * tf.cos(out_angle), out_mag * tf.sin(out_angle))
                outcomplex = self.specshift(outcomplex)
                outfea= tf.abs(tf.ifft2d(outcomplex))
                outfea = tf.expand_dims(outfea, -1)
                if j == 0:
                    outfeachannel = outfea
                else:
                    outfeachannel = tf.concat([outfeachannel, outfea], axis = -1)
            outfeachannel = tf.expand_dims(outfeachannel, 0)
            if i == 0:
                complexfea = outfeachannel
            else:
                complexfea = tf.concat([complexfea, outfeachannel], axis=0)
        return complexfea

    def transformGT(self,  GTdown):
        for i in range(self.batch_size):
            GT2 = GTdown[i, :, :, 0]
            GT_dft = tf.fft2d(tf.complex(GT2, tf.zeros_like(GT2)))
            GT_dft = self.specshift(GT_dft)
            GT_angle = tf.angle(GT_dft)
            GT_mag = tf.abs(GT_dft)
            GT_angle = tf.expand_dims(GT_angle, -1)
            GT_angle = tf.expand_dims(GT_angle, 0)
            GT_mag = tf.expand_dims(GT_mag, -1)
            GT_mag = tf.expand_dims(GT_mag, 0)
            if i == 0:
                GT_pha = GT_angle
                GT_amp = GT_mag
            else:
                GT_pha = tf.concat([GT_pha, GT_angle], axis=0)
                GT_amp = tf.concat([GT_amp, GT_mag], axis=0)
        return GT_pha, GT_amp

    def _loss(self, realmap, phamap, ampmap, finalmap, GT, amppattern):
        downshape = amppattern.shape
        self.phaloss = 0
        self.amploss = 0
        self.gtloss = 0
        self.realloss = 0
        self.phamap = self._normlized_0to1(phamap)
        self.ampmap = self._normlized_0to1(ampmap)
        GTdown = tf.image.resize_images(GT, downshape)
        GT_pha, GT_amp = self.transformGT( GTdown)
        self.amploss = tf.reduce_mean((ampmap - GT_amp) ** 2)
        self.phaloss = tf.reduce_mean((phamap - GT_pha) ** 2)
        norm_predict = self._normlized(finalmap)
        norm_GT = self._normlized(GT)
        self.gtloss = tf.reduce_sum(norm_GT * tf.log(self.eps + norm_GT / (norm_predict + self.eps)))
        norm_real = self._normlized(realmap)
        self.realloss = tf.reduce_sum(norm_GT * tf.log(self.eps + norm_GT / (norm_real + self.eps)))

        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)
        self.loss = (self.gtloss + loss_weight + self.factor_real * self.realloss + self.factor_angle * self.phaloss + self.factor_mag * self.amploss) / self.batch_size
        tf.summary.scalar('lossAll', self.loss)
        tf.summary.scalar('lossGT', self.gtloss / self.batch_size)
        tf.summary.scalar('lossReal', self.realloss / self.batch_size)
        #tf.summary.scalar('lossWeight', loss_weight / self.batch_size)
        tf.summary.scalar('lossPhase', self.factor_angle * self.phaloss / self.batch_size)
        tf.summary.scalar('lossAmplitude', self.factor_mag * self.amploss / self.batch_size)
        return self.loss


    def inference(self, input, seed, amppattern):
        self.ranseed = seed
        input_shapes = input.get_shape().as_list()
        self.batch_size = input_shapes[0]
        with tf.variable_scope('inference'):
            complexfea, nb_filter = self.encoder_net(input)
            fea_dim = int(complexfea.get_shape().as_list() [-1]/ 2)
            pro = self.ComplexProjectLayer('project', complexfea, pretrain=self.pretrain,
                                           trainable=self.trainable, use_bias=self.use_bias, type=self.prohecttype)
            feacomplex = tf.complex(complexfea[..., :fea_dim], complexfea[..., fea_dim:])
            feaphase = tf.angle(feacomplex)
            feaamplitude = tf.abs(feacomplex)
            realfeature, realmap = self.onstreamDecoder('realdecoder', pro, nb_filter, fuseindex = self.fusion_index, blocktype=[1, 1, 2, 2], multifusiion=False)
            amplitudefea,ampmap = self.onstreamDecoder('ampdecoder', feaamplitude, nb_filter, fuseindex = self.fusion_index, blocktype=[1, 1, 1, 1], multifusiion=False)
            phasefea,phamap = self.onstreamDecoder('phadecoder', feaphase, nb_filter, fuseindex = self.fusion_index, blocktype=[1, 1, 1, 1], multifusiion=False)
            realmap = self._normlized_0to1(realmap)
            phamap = self._normlized_0to1(phamap) * 2 * np.pi - np.pi
            amppattern = tf.expand_dims(amppattern, -1)
            amppattern = tf.expand_dims(amppattern, 0)
            amppattern = tf.cast(amppattern, tf.float32)
            ampmap = ampmap + amppattern
            realfea_shape = realfeature.get_shape().as_list()
            complexfea = self.feaIDFT(phasefea, amplitudefea)
            complexfea = tf.image.resize_images(complexfea, (realfea_shape[1], realfea_shape[2]))

            fusionfeatures = tf.concat([realfeature, complexfea], axis=-1)
            nb_filter2 = int(realfea_shape[-1]) * 2
            finalfea, finalmap = self.onstreamDecoder('finaldecoder', fusionfeatures, nb_filter2, fuseindex=self.fusion_index,
                                                        blocktype=[1, 1, 2, 2], multifusiion=False)
            finalmap = self._normlized_0to1(finalmap)
        return realmap, ampmap, phamap, finalmap



    def _train(self):
        opt = tf.train.AdamOptimizer(self.init_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
        # grads = opt.compute_gradients(self.loss, var_list=self.trainable_var_collection, aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grads = opt.compute_gradients(self.loss, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)
        # apply_gradient_op = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.loss)
        self.train = apply_gradient_op
        return apply_gradient_op

