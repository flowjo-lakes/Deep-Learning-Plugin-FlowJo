'''
Created on Dec 5, 2016

@author: urishaham
'''

import os.path
import errno
import keras.optimizers
from Calibration_Util import DataHandler as dh 
from Calibration_Util import FileIO as io
from keras.layers import Input, Dense, merge, Activation, add
from keras.models import Model
from keras import callbacks as cb
import numpy as np
import matplotlib
from keras.layers.normalization import BatchNormalization
#detect display
import os
havedisplay = "DISPLAY" in os.environ
#if we have a display use a plotting backend
if havedisplay:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')

import CostFunctions as cf
import Monitoring as mn
from keras.regularizers import l2
from sklearn import decomposition
from keras.callbacks import LearningRateScheduler
import math
import ScatterHist as sh
from keras import initializers
from numpy import genfromtxt
import sklearn.preprocessing as prep
import tensorflow as tf
import keras.backend as K
import sys
import os
import csv

# configuration hyper parameters
denoise = False # whether or not to train a denoising autoencoder to remove the zeros
keepProb=.8



####################
# CSV reformatting #
####################
def formatCSV( path ):
    inputFile = path
    noHeader = path + "/../swapFile.csv"
    result = path + "/../Result.csv"

    #get the column headers
    with open(inputFile) as f:
        reader = csv.reader(f, delimiter=',')
        global columnHeaders
        columnHeaders = next(reader)
        num_cols = len(columnHeaders)
        global col_index
        col_index = num_cols-1

    #delete the column headers, save the resultant in a different file
    with open(inputFile,'r') as f:
        with open(noHeader,'w') as f1:
            f.readline() # skip header line
            for line in f:
                f1.write(line)

    #get the row identifiers
    with open(noHeader, 'r') as f:
        reader = csv.reader(f)
        global rowIdentifiers
        rowIdentifiers = []
        for row in reader:
            content = list(row[i] for i in [col_index])
            rowIdentifiers.append(content)

    #delete the row identifiers
    with open(noHeader, "r") as fp_in, open(result, "w", encoding="UTF-8") as fp_out:
        reader = csv.reader(fp_in)
        writer = csv.writer(fp_out, lineterminator='\n')
        for row in reader:
            del row[col_index]
            writer.writerow(row)

    # clean up extraneous file...
    os.remove(noHeader)

    # At this point the file is ready to be processed by the deep learning algorithm
    return result

#read a list of arguments passed from the command line
numEpochs = int(sys.argv[1])
targetPath = formatCSV( sys.argv[3] )
sourcePath = formatCSV( sys.argv[2] )
outputFolder = sys.argv[4]
resultName = sys.argv[5]
resultPath = sourcePath + "/../_" + resultName + "_" + str(numEpochs) + "_Epochs.csv"
realResultPath = outputFolder + "/" + resultName + "_" + str(numEpochs) + "E_" + "DL.csv"

# AE confiduration
ae_encodingDim = 25
l2_penalty_ae = 1e-2 

#MMD net configuration
mmdNetLayerSizes = [25, 25]
l2_penalty = 1e-2
#init = lambda shape, name:initializations.normal(shape, scale=.1e-4, name=name)
#def my_init (shape):
#    return initializers.normal(stddev=.1e-4)
#my_init = 'glorot_normal'

#######################
###### read data ######
#######################

#might have to worry about using headers in csv files
source = genfromtxt(sourcePath, delimiter=',', skip_header=0)
target = genfromtxt(targetPath, delimiter=',', skip_header=0)

# pre-process data: log transformation, a standard practice with CyTOF data
target = dh.preProcessCytofData(target)
source = dh.preProcessCytofData(source) 

numZerosOK=1
toKeepS = np.sum((source==0), axis = 1) <=numZerosOK
print(np.sum(toKeepS))
toKeepT = np.sum((target==0), axis = 1) <=numZerosOK
print(np.sum(toKeepT))

inputDim = target.shape[1]

#It would appear that nothing about this if statement needs to be modified to work with the plugin
if denoise:
    trainTarget_ae = np.concatenate([source[toKeepS], target[toKeepT]], axis=0)
    np.random.shuffle(trainTarget_ae)
    trainData_ae = trainTarget_ae * np.random.binomial(n=1, p=keepProb, size = trainTarget_ae.shape)
    #define the model 5 lines
    input_cell = Input(shape=(inputDim,))
    encoded = Dense(ae_encodingDim, activation='relu',W_regularizer=l2(l2_penalty_ae))(input_cell)
    encoded1 = Dense(ae_encodingDim, activation='relu',W_regularizer=l2(l2_penalty_ae))(encoded)
    decoded = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty_ae))(encoded1)
    autoencoder = Model(input=input_cell, output=decoded)
    autoencoder.compile(optimizer='rmsprop', loss='mse')
    #A few options here that may be relevant to change, might be useful to ask creator
    #more training happens here
    autoencoder.fit(trainData_ae, trainTarget_ae, epochs=numEpochs, batch_size=128, shuffle=True,  validation_split=0.1,
                    callbacks=[mn.monitor(), cb.EarlyStopping(monitor='val_loss', patience=25,  mode='auto')])    
    source = autoencoder.predict(source)
    target = autoencoder.predict(target)

# rescale source to have zero mean and unit variance
# apply same transformation to the target
preprocessor = prep.StandardScaler().fit(source)
source = preprocessor.transform(source) 
target = preprocessor.transform(target)    

#############################
######## train MMD net ######
#############################


calibInput = Input(shape=(inputDim,))
block1_bn1 = BatchNormalization()(calibInput)
block1_a1 = Activation('relu')(block1_bn1)
block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a1) 
block1_bn2 = BatchNormalization()(block1_w1)
block1_a2 = Activation('relu')(block1_bn2)
block1_w2 = Dense(inputDim, activation='linear',kernel_regularizer=l2(l2_penalty),
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a2) 
block1_output = add([block1_w2, calibInput])
block2_bn1 = BatchNormalization()(block1_output)
block2_a1 = Activation('relu')(block2_bn1)
block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a1) 
block2_bn2 = BatchNormalization()(block2_w1)
block2_a2 = Activation('relu')(block2_bn2)
block2_w2 = Dense(inputDim, activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a2) 
block2_output = add([block2_w2, block1_output])
block3_bn1 = BatchNormalization()(block2_output)
block3_a1 = Activation('relu')(block3_bn1)
block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a1) 
block3_bn2 = BatchNormalization()(block3_w1)
block3_a2 = Activation('relu')(block3_bn2)
block3_w2 = Dense(inputDim, activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a2) 
block3_output = add([block3_w2, block2_output])

calibMMDNet = Model(inputs=calibInput, outputs=block3_output)

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 150.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)

#train MMD net
optimizer = keras.optimizers.rmsprop(lr=0.0)
#training occurs here
calibMMDNet.compile(optimizer=optimizer, loss=lambda y_true,y_pred: 
               cf.MMD(block3_output,target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))
K.get_session().run(tf.global_variables_initializer()) #wipes out weights

sourceLabels = np.zeros(source.shape[0])
calibMMDNet.fit(source,sourceLabels,nb_epoch=numEpochs,batch_size=1000,validation_split=0.1,verbose=1,
           callbacks=[lrate, mn.monitorMMD(source, target, calibMMDNet.predict),
                      cb.EarlyStopping(monitor='val_loss',patience=50,mode='auto')])

##############################
###### evaluate results ######
##############################

#can call this file on other samples
calibratedSource = calibMMDNet.predict(source)
#np.savetxt(resultPath, calibratedSource, delimiter=",")

#################################### qualitative evaluation: PCA #####################################
pca = decomposition.PCA()
pca.fit(target)

# project data onto PCs
target_sample_pca = pca.transform(target)
projection_before = pca.transform(source)
projection_after = pca.transform(calibratedSource)

# choose PCs to plot
pc1 = 0
pc2 = 1
axis1 = 'PC'+str(pc1)
axis2 = 'PC'+str(pc2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_before[:,pc1], projection_before[:,pc2], axis1, axis2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after[:,pc1], projection_after[:,pc2], axis1, axis2)

#use the following code block if we want to save weight files from training sessions
#if denoise:
    #autoencoder.save(os.path.join(io.DeepLearningRoot(), "savedModels/person1_baseline_DAEL.h5"))

#calibMMDNet.save_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_ResNet_weights.h5'))

#save the calibrated file
np.savetxt(resultPath, calibratedSource, delimiter=",")
resultantFile = resultPath

######################################
# Reformat csv back to SeqGeq format #
######################################

#read back the resultant data
with open(resultantFile, 'r') as f:
    reader = csv.reader(f)
    bareData = f.readlines()

#add the headers and row identifiers back to the data
with open(resultantFile, 'w') as f:
    wr = csv.writer(f, lineterminator='\n', delimiter=',')
    wr.writerow(columnHeaders)
    reader = csv.reader(bareData)
    for list in rowIdentifiers:
        wr.writerow(next(reader) + list)

with open(resultantFile, 'r') as infile, open(realResultPath, 'w') as outfile:
        data = infile.read()
        data = data.replace("e+00", "")
        data = data.replace("+", "")
        outfile.write(data)

#clean up extraneous file
os.remove(sourcePath)
os.remove(resultantFile)

