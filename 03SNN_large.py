# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:12:43 2019
@author: anonymous
"""
#%tensorflow_version 1.2
from __future__ import print_function

import argparse
parser = argparse.ArgumentParser(description='TNN runner')
parser.add_argument('--dataset', help='Options are: bostonHousing, concreteData, energyEfficiency, proteinStructure, randomFunction', dest='dataset', default='bostonHousing')
parser.add_argument('--dataset_path', help='Relative path to datasets', dest='dataset_path', default='./data/')
parser.add_argument('--val_pct', help='Percentage of validation split.', dest='val_pct', default=0.05, type=float)
parser.add_argument('--test_pct', help='Percentage of test split.', dest='test_pct', default=0.05, type=float)
parser.add_argument('--l2', help='L2 Regularization weighting.', dest='l2', default=0.0, type=float)
parser.add_argument('--seed', help='Random seed', dest='seed', default=13, type=int)
parser.add_argument('--num', help='Number of datapoints for random function', default=1000, type=int)
parser = parser.parse_args()

# set the seed
print('Random seed:', parser.seed)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(parser.seed)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(parser.seed)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_random_seed(parser.seed)

# for later versions: 

from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
import tensorflow.keras
import itertools
from data.data import *
import sys
from progressbar import ProgressBar as PB
from SNN_helper import *
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Subtract,Input,Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau,EarlyStopping
from sklearn.linear_model import LinearRegression

### scaling factor 100,300,1000,3000,10000,30000,100000
n=100000

#(x_full, y_full) = getData(parser.dataset, parser.dataset_path)
(x_full, y_full) = getData('randomFunction', './data/',n)
(x_train_single, y_train_single), (x_val_single, y_val_single), (x_test_single, y_test_single) = splitData(x_full, y_full, val_pct=parser.val_pct, test_pct=parser.test_pct)

##### center and normalize
   
cn_transformer=CenterAndNorm()    


x_train_single,y_train_single=cn_transformer.fittransform(x_train_single,y_train_single)
x_val_single,y_val_single=cn_transformer.transform(x_val_single,y_val_single)
x_test_single,y_test_single=cn_transformer.transform(x_test_single,y_test_single)

##########################################


############ NN parameters

batch_size = 16
epochs = 10000

############### create NN

observer_a=Input(shape=(x_train_single.shape[-1],),name='observer_a')
observer_b=Input(shape=(x_train_single.shape[-1],),name='observer_b')

l2=parser.l2

merged_layer = tensorflow.keras.layers.Concatenate()([observer_a, observer_b])

merged_layer=Dense(192,activation='relu',kernel_regularizer=regularizers.l2(l2),activity_regularizer=regularizers.l2(l2))(merged_layer)
merged_layer=Dense(192,activation='relu',name='iout',kernel_regularizer=regularizers.l2(l2),activity_regularizer=regularizers.l2(l2))(merged_layer)
output=Dense(1,kernel_regularizer=regularizers.l2(l2),activity_regularizer=regularizers.l2(l2))(merged_layer)
model = Model(inputs=[observer_a, observer_b], outputs=output)
model.summary()

# Let's train the model 
model.compile(loss='mse'
              ,optimizer=tensorflow.keras.optimizers.Adadelta(lr=1)
              #,optimizer='rmsprop'
              ,metrics=['mse']
              )

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=int(1000/np.sqrt(n)+1), verbose=1,min_lr=0)


early_stop= EarlyStopping(monitor='val_loss', patience=3*int(1000/np.sqrt(n)+1), verbose=1)
mcp_save = ModelCheckpoint('mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
history = model.fit_generator(generator_sym(x_train_single, y_train_single, batch_size),
                                steps_per_epoch=len(x_train_single)*10/batch_size,
                                epochs=epochs,
                                validation_data=generator_sym(x_val_single, y_val_single, batch_size),
                                validation_steps=len(x_val_single)*100/batch_size,
                                callbacks=[reduce_lr,early_stop, mcp_save],verbose=1)

model.load_weights('mdl_wts.hdf5')

# Plot training & test loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='upper left')
# # plt.show()
# plt.savefig('loss.pdf')
    

# Y_pred_train=[]
# Y_pred_r_train=[]
# Y_pred_check_train=[]
# Y_median_train=[]
# Y_var_train=[]
# Y_mse_train=[]
# 
# 
# for i in PB()(range(len(x_train_single))):
#     pair_B=np.array([x_train_single[i]]*len(x_train_single))
#     Y_pred_train.append(np.average(0.5*model.predict([pair_B,x_train_single]).flatten()-0.5*model.predict([x_train_single,pair_B]).flatten()+y_train_single))
#     Y_pred_r_train.append(np.average(-model.predict([x_train_single,pair_B]).flatten()+y_train_single))
#     Y_pred_check_train.append(np.var(0.5*model.predict([pair_B,x_train_single])+0.5*model.predict([x_train_single,pair_B])))
#     #Y_pred_check_train.append(np.average(np.abs(model.predict([pair_B,x_train_single])+model.predict([x_train_single,pair_B]))))
#     Y_median_train.append(np.median(model.predict([pair_B,x_train_single]).flatten()+y_train_single))
#     Y_var_train.append(np.var(0.5*model.predict([pair_B,x_train_single]).flatten()-0.5*model.predict([x_train_single,pair_B]).flatten()+y_train_single))
#     Y_mse_train.append((Y_pred_train[i]-y_train_single[i])**2)
#     
# 
# Y_pred_train=cn_transformer.inversetransformY(np.array(Y_pred_train))
# Y_pred_r_train=cn_transformer.inversetransformY(np.array(Y_pred_r_train))
# Y_pred_check_train=np.array(Y_pred_check_train)*(cn_transformer.Ymax)
# Y_median_train=cn_transformer.inversetransformY(np.array(Y_median_train))
# Y_var_train=np.array(Y_var_train)*(cn_transformer.Ymax)**2
# Y_mse_train=np.array(Y_mse_train)*(cn_transformer.Ymax)**2
# Y_self_check_train=np.abs(np.array(model.predict([x_train_single,x_train_single]))).flatten()*(cn_transformer.Ymax)

#####################

Y_pred_test=[]
Y_pred_r_test=[]
Y_pred_check_test=[]
Y_median_test=[]
Y_var_test=[]
Y_mse_test=[]



for i in PB()(range(len(x_test_single))):
    pair_B=np.array([x_test_single[i]]*len(x_train_single))
    diffA=model.predict([pair_B,x_train_single]).flatten()
    diffB=model.predict([x_train_single,pair_B]).flatten()
    Y_pred_test.append(np.average(0.5*diffA-0.5*diffB+y_train_single, weights=None))
    Y_pred_r_test.append(np.average(-diffB+y_train_single))
    Y_pred_check_test.append(np.var(0.5*diffA+0.5*diffB))
    #Y_pred_check_test.append(np.average(np.abs(model.predict([pair_B,x_train_single])+model.predict([x_train_single,pair_B]))))
    Y_median_test.append(np.median(diffA+y_train_single))
    Y_var_test.append(np.var(0.5*diffA-0.5*diffB+y_train_single))
    Y_mse_test.append((Y_pred_test[i]-y_test_single[i])**2)


Y_pred_test=cn_transformer.inversetransformY(np.array(Y_pred_test))
Y_pred_r_test=cn_transformer.inversetransformY(np.array(Y_pred_r_test))
Y_pred_check_test=np.array(Y_pred_check_test)*(cn_transformer.Ymax)
Y_median_test=cn_transformer.inversetransformY(np.array(Y_median_test))
Y_var_test=np.array(Y_var_test)*(cn_transformer.Ymax)**2
Y_mse_test=np.array(Y_mse_test)*(cn_transformer.Ymax)**2
Y_self_check_test=np.abs(np.array(model.predict([x_test_single,x_test_single]))).flatten()*(cn_transformer.Ymax)
#####################
    
# Y_pred_val=[]
# Y_pred_r_val=[]
# Y_pred_check_val=[]
# Y_median_val=[]
# Y_var_val=[]
# Y_mse_val=[]
# 
# 
# 
# for i in PB()(range(len(x_val_single))):
#     pair_B=np.array([x_val_single[i]]*len(x_train_single))
#     Y_pred_val.append(np.average(0.5*model.predict([pair_B,x_train_single]).flatten()-0.5*model.predict([x_train_single,pair_B]).flatten()+y_train_single))
#     Y_pred_r_val.append(np.average(-model.predict([x_train_single,pair_B]).flatten()+y_train_single))
#     Y_pred_check_val.append(np.var(0.5*model.predict([pair_B,x_train_single])+0.5*model.predict([x_train_single,pair_B])))
#     #Y_pred_check_val.append(np.average(np.abs(model.predict([pair_B,x_train_single])+model.predict([x_train_single,pair_B]))))
#     Y_median_val.append(np.median(model.predict([pair_B,x_train_single]).flatten()+y_train_single))
#     Y_var_val.append(np.var(0.5*model.predict([pair_B,x_train_single]).flatten()-0.5*model.predict([x_train_single,pair_B]).flatten()+y_train_single))
#     Y_mse_val.append((Y_pred_val[i]-y_val_single[i])**2)
# 
# 
# Y_pred_val=cn_transformer.inversetransformY(np.array(Y_pred_val))
# Y_pred_r_val=cn_transformer.inversetransformY(np.array(Y_pred_r_val))
# Y_pred_check_val=np.array(Y_pred_check_val)*(cn_transformer.Ymax)
# Y_median_val=cn_transformer.inversetransformY(np.array(Y_median_val))
# Y_var_val=np.array(Y_var_val)*(cn_transformer.Ymax)**2
# Y_mse_val=np.array(Y_mse_val)*(cn_transformer.Ymax)**2
# Y_self_check_val=np.abs(np.array(model.predict([x_val_single,x_val_single]))).flatten()*(cn_transformer.Ymax)
#####################


# print('Train RMSE:', np.average(Y_mse_train)**0.5)
# print('Val RMSE:',np.average(Y_mse_val)**0.5)
print('Test RMSE:',np.average(Y_mse_test)**0.5)
