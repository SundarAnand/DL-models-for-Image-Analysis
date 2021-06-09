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


def DenseNet(input_shape, dense_blocks=3, dense_layers=4, growth_rate=12, nb_classes=2, dropout_rate=None,
             bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40):

    """
    Creating a DenseNet
    
    Arguments:
        input_shape  : shape of the input images. E.g. (28,28,1) for MNIST    
        dense_blocks : amount of dense blocks that will be created (default: 3)    
        dense_layers : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
                       or define only 2 to add 2 layers at all dense blocks. -1 means that dense_layers will be calculated
                       by the given depth (default: -1)
        growth_rate  : number of filters to add per dense block (default: 12)
        nb_classes   : number of classes
        dropout_rate : defines the dropout rate that is accomplished after each conv layer (except the first one).
                       In the paper the authors recommend a dropout of 0.2 (default: None)
        bottleneck   : (True / False) if true it will be added in convolution block (default: False)
        compression  : reduce the number of feature-maps at transition layer. In the paper the authors recomment a compression
                       of 0.5 (default: 1.0 - will have no compression effect)
        weight_decay : weight decay of L2 regularization on weights (default: 1e-4)
        depth        : number or layers (default: 40)
        
    Returns:
        Model        : A Keras model instance
    """
    
    if nb_classes==None:
        raise Exception('Please define number of classes (e.g. num_classes=10). This is required for final softmax.')
    
    if compression <=0.0 or compression > 1.0:
        raise Exception('Compression have to be a value between 0.0 and 1.0. If you set compression to 1.0 it will be turn off.')
    
    if type(dense_layers) is list:
        if len(dense_layers) != dense_blocks:
            raise AssertionError('Number of dense blocks have to be same length to specified layers')
    elif dense_layers == -1:
        if bottleneck:
            dense_layers = (depth - (dense_blocks + 1))/dense_blocks // 2
        else:
            dense_layers = (depth - (dense_blocks + 1))//dense_blocks
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
    else:
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
        
    img_input = Input(shape=input_shape)
    nb_channels = growth_rate * 2
    
    print('Creating DenseNet')
    print('#############################################')
    print('Dense blocks: %s' % dense_blocks)
    print('Layers per dense block: %s' % dense_layers)
    print('#############################################')
    
    # Initial convolution layer
    x = Conv2D(nb_channels, (3,3), padding='same',strides=(1,1),
                      use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
    
    # Building dense blocks
    for block in range(dense_blocks):
        
        # Add dense block
        x, nb_channels = dense_block(x, dense_layers[block], nb_channels, growth_rate, dropout_rate, bottleneck, weight_decay)
        
        if block < dense_blocks - 1:  # if it's not the last dense block
            # Add transition_block
            x = transition_layer(x, nb_channels, dropout_rate, compression, weight_decay)
            nb_channels = int(nb_channels * compression)
    
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
    
    model_name = None
    if growth_rate >= 36:
        model_name = 'widedense'
    else:
        model_name = 'dense'
        
    if bottleneck:
        model_name = model_name + 'b'
        
    if compression < 1.0:
        model_name = model_name + 'c'
        
    return Model(img_input, x, name=model_name)
    
def dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    """
    Creates a dense block and concatenates inputs
    """
    
    x_list = [x]
    for i in range(nb_layers):
        cb = convolution_block(x, growth_rate, dropout_rate, bottleneck, weight_decay)
        x_list.append(cb)
        x = Concatenate(axis=-1)(x_list)
        nb_channels += growth_rate
    return x, nb_channels


def convolution_block(x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    """
    Creates a convolution block consisting of BN-ReLU-Conv.
    Optional: bottleneck, dropout
    """
    
    # Bottleneck
    if bottleneck:
        bottleneckWidth = 4
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_channels * bottleneckWidth, (1, 1), use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        # Dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
    
    # Standard (BN-ReLU-Conv)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    
    # Dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    return x


def transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):
    """
    Creates a transition layer between dense blocks as transition, which do convolution and pooling.
    Works as downsampling.
    """
    
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_channels*compression), (1, 1), padding='same',
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    
    # Adding dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x


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
        model = DenseNet((256, 256, 3))
        ## Compling the model
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
        model.summary()
        history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=50)
        final_list = []
        for i in range(0, len(history.history['accuracy'])):
            final_list.append([i+1, history.history['accuracy'][i], history.history['loss'][i], history.history['f1_m'][i], history.history['precision_m'][i], history.history['recall_m'][i], history.history['val_accuracy'][i], history.history['val_loss'][i], history.history['val_f1_m'][i], history.history['val_precision_m'][i], history.history['val_recall_m'][i]])
        df_output = pd.DataFrame(final_list)
        df_output.columns = ["epoch", "accuracy", "loss", "f1_m", "precision_m", "recall_m", "val_accuracy", "val_loss", "val_f1_m", "val_precision_m", "val_recall_m"]
        df_output.to_csv("DenseNet_Fold_" + str(count) + ".csv", index=False)
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
    df_epochs.to_csv("DenseNetResults.csv", index=False)
    model.save('DenseNet.h5')
