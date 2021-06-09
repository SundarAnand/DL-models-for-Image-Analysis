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
from keras import applications
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


def ResNet ():
    base_model = applications.resnet.ResNet101(weights= None, include_top=False, input_shape= (256,256,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(2, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
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
        model = ResNet()
        ## Compling the model
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
        model.summary()
        history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=50)
        final_list = []
        for i in range(0, len(history.history['accuracy'])):
            final_list.append([i+1, history.history['accuracy'][i], history.history['loss'][i], history.history['f1_m'][i], history.history['precision_m'][i], history.history['recall_m'][i], history.history['val_accuracy'][i], history.history['val_loss'][i], history.history['val_f1_m'][i], history.history['val_precision_m'][i], history.history['val_recall_m'][i]])
        df_output = pd.DataFrame(final_list)
        df_output.columns = ["epoch", "accuracy", "loss", "f1_m", "precision_m", "recall_m", "val_accuracy", "val_loss", "val_f1_m", "val_precision_m", "val_recall_m"]
        df_output.to_csv("ResNet_Fold_" + str(count) + ".csv", index=False)
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
    df_epochs.to_csv("ResNetResults.csv", index=False)
    model.save('ResNet.h5')
