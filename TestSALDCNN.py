
import numpy as np
import tensorflow as tf
#import time
import random
import ComplexDenseNet4 as Network
import glob
import os
#from PIL import Image
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ModelName = './model/SALDCNN_eIDFT-46'#
OutputDir = './result/'
InputDir = './img/'

netcompress = 0.5 # compression factor
netgr = 48 # growth_rate
netblock = [6, 12, 32] # numbers of conv layer for dense block
inputMethod = 'realcomplexRGB' # Determine you input channels, you can choose: complexgray, commplexRGB, real, realcomplexgray, realcomplexRGB

random.seed(a=730)
tf.set_random_seed(730)
ranseed = 730
batch_size = 1

input_size = [240, 320]
output_size = [input_size[0]/4, input_size[1]/4]
output_size_TD = [15, 20]
amppattern =  np.load('amppattern2_' + str(output_size_TD[0]) + '_' + str(output_size_TD[1])+'.npz')["arr_0"]
meanR = 103.939
meanG = 116.779
meanB = 123.68
meanamp = np.mean(amppattern)
maxamp = np.max(amppattern)



if inputMethod == 'complexgray':
    inputchannal = 1
elif inputMethod == 'commplexRGB':
    inputchannal = 3
elif inputMethod == 'real':
    inputchannal = 3
elif inputMethod == 'realcomplexgray':
    inputchannal = 4
elif inputMethod == 'realcomplexRGB':
    inputchannal = 6



def PreProcess(input, method='complexgray'):  # complexgray, commplexRGB, real, realcomplexRGB, realcomplexRGB
    if method == 'complexgray':
        input = input[..., np.newaxis]
        input = input / 255.0
        gray = input[..., 0,:] * 0.299 + input[..., 1,:] * 0.587 + input[..., 2,:] * 0.114
        outreal, outimag = GetSpectrum(gray)
        output = np.concatenate([outreal, outimag], axis=-1)
    elif method == 'commplexRGB':
        input = input / 255.0
        input = input[..., np.newaxis]
        Rchannel = input[..., 0, :]
        Gchannel = input[..., 1, :]
        Bchannel = input[..., 2, :]
        Routreal, Routimag = GetSpectrum(Rchannel)
        Goutreal, Goutimag = GetSpectrum(Gchannel)
        Boutreal, Boutimag = GetSpectrum(Bchannel)
        output = np.concatenate([Routreal, Goutreal,Boutreal,Routimag,Goutimag,Boutimag], axis=-1)
    elif method == 'real':
        input = input[..., np.newaxis]
        Rchannel = (input[..., 0, :] - meanR)/128
        Gchannel = (input[..., 1, :] - meanG)/128
        Bchannel = (input[..., 2, :] - meanB)/128
        RGB = np.concatenate([Rchannel, Gchannel,Bchannel], axis=-1)
        output = np.concatenate([RGB,  np.zeros_like(RGB)], axis=-1)
    elif method == 'realcomplexgray':
        RGB = input / 255.0
        input = input[..., np.newaxis]
        input = input / 255.0
        gray = input[..., 0, :] * 0.299 + input[..., 1, :] * 0.587 + input[..., 2, :] * 0.114
        outreal, outimag = GetSpectrum(gray)
        output = np.concatenate([RGB, outreal, np.zeros_like(RGB),outimag], axis=-1)
    elif method == 'realcomplexRGB':
        RGB = input / 255.0
        input = input[..., np.newaxis]
        input = input / 255.0
        Rchannel = input[..., 0, :]
        Gchannel = input[..., 1, :]
        Bchannel = input[..., 2, :]
        Routreal, Routimag = GetSpectrum(Rchannel)
        Goutreal, Goutimag = GetSpectrum(Gchannel)
        Boutreal, Boutimag = GetSpectrum(Bchannel)
        output = np.concatenate([RGB, Routreal, Goutreal, Boutreal,np.zeros_like(RGB), Routimag, Goutimag, Boutimag], axis=-1)
    return output

def main():
    net = Network.Net()
    inputs = tf.placeholder(tf.float32, (batch_size, input_size[0], input_size[1], inputchannal*2))
    # GroundTruth = tf.placeholder(tf.float32, (batch_size, output_size[0], output_size[1],1))

    net.encoder_compress = netcompress
    net.growth_rate = netgr
    net.blockfilers = netblock
    realmap, ampmap, phamap, finalmap = net.inference(inputs, ranseed, amppattern)  # inference
    #loss_op = net._loss(realmap, phamap, ampmap, finalmap, amppattern)  # loss
    preidct_op = finalmap

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, ModelName)

    imglist =  glob.glob(InputDir + '*.jpg')
    for imagePath in  imglist:
        full_name = os.path.split(imagePath)[-1]
        (imgname, file_type) = os.path.splitext(full_name)
        print('Processing %s'%(imagePath))
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_shape = [image.shape[0], image.shape[1]]
        image = cv2.resize(image,(input_size[1], input_size[0]))
        image = image[np.newaxis, ...]
        image = image.astype(np.float32)
        if image.ndim == 2:
            image = image[..., np.newaxis]
            image = np.concatenate((image, image, image), axis=-1)
        input = PreProcess(image, method=inputMethod)
        np_predict = sess.run(preidct_op, feed_dict={inputs: input})
        np_predict = np_predict[0, :, :, 0]
        Out_frame = cv2.resize(np_predict, (ori_shape[1], ori_shape[0]))
        Out_frame = Out_frame * 255
        Out_frame = np.uint8(Out_frame)
        cv2.imwrite(OutputDir + imgname + '.jpg', Out_frame)


def cacCC(gtsAnn, resAnn):
    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)
    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]

def cacKL(gtsAnn, resAnn,eps = 1e-7):
    if np.sum(gtsAnn) > 0:
        gtsAnn = gtsAnn / np.sum(gtsAnn)
    if np.sum(resAnn) > 0:
        resAnn = resAnn / np.sum(resAnn)
    return np.sum(gtsAnn * np.log(eps + gtsAnn / (resAnn + eps)))

def batchresize(input, newsize):
    batchsize = input.shape[0]
    for i in range(batchsize):
        temp = input[i,...]
        outtemp = cv2.resize(temp, newsize)
        outtemp = outtemp[np.newaxis, ...]
        if i == 0:
            output = outtemp
        else:
            output = np.concatenate([output, outtemp], axis=0)
    return output

def GetSpectrum(input):
    resizedfea = batchresize(input, (output_size_TD[1], output_size_TD[0]))
    resizedfea = SpeShift(resizedfea)
    fftfea = np.fft.fft2(resizedfea, axes=(1, 2)) / maxamp
    outreal = batchresize(np.real(fftfea), (input_size[1], input_size[0]))
    outimag = batchresize(np.imag(fftfea), (input_size[1], input_size[0]))
    outreal = outreal[..., np.newaxis]
    outimag = outimag[..., np.newaxis]
    return outreal, outimag

def SpeShift(input):
    ishape = input.shape
    assert len(ishape) == 3
    h = ishape[1]
    w = ishape[2]
    p1 = input[...,0: h // 2, 0: w // 2]
    p2 = input[...,h // 2: h, 0: w // 2]
    p3 = input[...,0: h // 2, w // 2: w]
    p4 = input[...,h // 2: h, w // 2: w]
    p13 = np.concatenate((p3,p1),2)
    p24 = np.concatenate((p4, p2),2)
    pout = np.concatenate((p24, p13),1)
    return pout

if __name__ == '__main__':
    main()