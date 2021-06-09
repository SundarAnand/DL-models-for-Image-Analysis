import keras
from keras.layers import merge, Conv2D, Input, Reshape, Activation
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.utils.vis_utils import plot_model
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from numpy import genfromtxt
from keras.preprocessing.image import img_to_array as img_to_array
from keras.preprocessing.image import load_img as load_img
from keras import backend as K
from sklearn.metrics import classification_report
import seaborn as sns
from keras.layers import Input, Dense, Dropout, Activation, Concatenate, BatchNormalization
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K

# essentials
import numpy as np
import math
import os
import random
import pandas as pd
import tensorflow as tf
import argparse
from matplotlib import pyplot as plt
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support

# tensorflow, keras
import keras
from keras.layers import merge, Conv2D, Input, Reshape, Activation, Add, Multiply
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.utils.vis_utils import plot_model
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from numpy import genfromtxt
from keras.preprocessing.image import img_to_array as img_to_array
from keras.preprocessing.image import load_img as load_img
from sklearn.model_selection import KFold 

# essentials
import numpy as np
import math
import os
import random
import pandas as pd
import tensorflow as tf
import argparse
# from matplotlib.pyplot import pyplot as plt
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# attention needed
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Reshape
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import random
import sys
from numpy import load
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, Conv2D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D,Input, Dense
from keras.layers import MaxPool2D,AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x

def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1):

    bn_axis = -1

    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret

def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                      strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1,
                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))

    return x

def ShuffleNetV2(input_shape, include_top=True, input_tensor=None, scale_factor=1.0, pooling='max', num_shuffle_units=[3,7,3], bottleneck_ratio=1, classes=2):
    
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))

    input_shape = input_shape

    out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}

    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  #  calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # create shufflenet architecture
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                   repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   stage=stage + 2)

    if bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048
    x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)

    if include_top:
        x = Dense(classes, name='fc')(x)
        x = Activation('softmax', name='softmax')(x)

    if input_tensor:
        inputs = get_source_inputs(input_tensor)

    else:
        inputs = img_input

    model = Model(inputs, x, name=name)

    return model

# load train and test dataset
def load_dataset():
	# load dataset
	data = load('COVID.npz')
	x, y = data['arr_0'], data['arr_1']
	print(x.shape, y.shape)
	return x, y

def val_split(X, Y):
    trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.25, random_state=1)
    print(trainX.shape, trainY.shape, valX.shape, valY.shape)
    return trainX, trainY, valX, valY 


##Using the GPU

with tf.device('/device:GPU:0'):

    # Defined shared layers as global variables
    concatenator = Concatenate(axis=-1)
    densor_block2 = Dense(1, activation="relu")
    densor_block3 = Dense(1, activation="relu")
    activator = Activation('softmax', name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes=2)
    mult = Multiply()

    ##Loading the data
    X, y = load_dataset()

    ##5 Fold Validation
    kf = KFold(n_splits=5)
    count = 0
    metrices = []

    for train_index, test_index in kf.split(X, y):
        count = count + 1
        print ("Fold : " + str(count))
        x_train, y_train = X[train_index], y[train_index]
        testX, testY = X[test_index], y[test_index]
        trainX, trainY, valX, valY = val_split(x_train, y_train)
        ## Initializing the model
        model = ShuffleNetV2((256, 256, 3))
        ## Compling the model
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
        model.summary()
        history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=50)
        final_list = []
        for i in range(0, len(history.history['accuracy'])):
            final_list.append([i+1, history.history['accuracy'][i], history.history['loss'][i], history.history['f1_m'][i], history.history['precision_m'][i], history.history['recall_m'][i], history.history['val_accuracy'][i], history.history['val_loss'][i], history.history['val_f1_m'][i], history.history['val_precision_m'][i], history.history['val_recall_m'][i]])
        df_output = pd.DataFrame(final_list)
        df_output.columns = ["epoch", "accuracy", "loss", "f1_m", "precision_m", "recall_m", "val_accuracy", "val_loss", "val_f1_m", "val_precision_m", "val_recall_m"]
        df_output.to_csv("ShuffleNet_Fold_" + str(count) + ".csv", index=False)
        testY_bool = np.argmax(testY, axis=1)
        y_pred = model.predict(testX, batch_size=64, verbose=1)
        y_pred_bool = np.argmax(y_pred, axis=1)
        confusion = confusion_matrix(testY_bool, y_pred_bool) 
        print ('Confusion Matrix :')
        print(confusion) 
        print ('Report : ')
        print(classification_report(testY_bool, y_pred_bool))
        print ("Accuracy : ")
        print (accuracy_score(testY_bool, y_pred_bool))
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        print ("Accuracy from confusion matrix : ")
        print ((TP + TN) / float(TP + TN + FP + FN))
        sensitivity = TP / float(FN + TP)
        print("Sensitivity : ")
        print(sensitivity)
        print ("Recall : ")
        print(recall_score(testY_bool, y_pred_bool))
        specificity = TN / (TN + FP)
        print("Specificity")
        print(specificity)
        print("Precision : ")
        print(precision_score(testY_bool, y_pred_bool))
        print("F1_Score : ")
        print(f1_score(testY_bool, y_pred_bool))
        metrices.append([count, accuracy_score(testY_bool, y_pred_bool), recall_score(testY_bool, y_pred_bool), specificity, precision_score(testY_bool, y_pred_bool), f1_score(testY_bool, y_pred_bool)])

    df_epochs = pd.DataFrame(metrices)
    df_epochs.columns = ["Fold", "Test Accuracy", "Recall", "Specificity", "Precision", "F1 Score"]
    df_epochs.to_csv("ShuffleNetResults.csv", index=False)
    model.save('ShuffleNet.h5')
