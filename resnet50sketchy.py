# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:49:49 2019

@author: ushasi2
help:
https://keras.io/applications/#resnet
"""
#import matplotlib.pyplot as plt
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout
import scipy.io as sio
import numpy as np
import os

NUM_CLASSES = 125		####
CHANNELS = 3
IMAGE_RESIZE = 224	####
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 8
#STEPS_PER_EPOCH_TRAINING = 10
#STEPS_PER_EPOCH_VALIDATION = 10
BATCH_SIZE_TRAINING = 64	####
BATCH_SIZE_VALIDATION = 64	####
BATCH_SIZE_TESTING = 64


os.environ["CUDA_VISIBLE_DEVICES"]="0"
resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = Sequential()
model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = resnet_weights_path)) #None
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))
#model.add(Dropout(0.3))
#model.layers[0].trainable = True
model.summary()

for idx, layer in enumerate(model.layers):
        print(idx,layer.name, layer.trainable)

sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

image_size = IMAGE_RESIZE


data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split = 0.2)	####

train_generator = data_generator.flow_from_directory(
        'sketch/tx_000000000000',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_TRAINING,
        class_mode='categorical',
	subset = 'training')


validation_generator = data_generator.flow_from_directory(
        'sketch/tx_000000000000',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_VALIDATION,
        class_mode='categorical',
	subset = 'validation') 

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = 'best_image.hdf5', monitor = 'val_acc', save_best_only = True, mode = 'auto')
print('hi')
fit_history = model.fit((train_generator),
        #steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
        epochs = NUM_EPOCHS,
        validation_data=(validation_generator),
        #validation_steps=STEPS_PER_EPOCH_VALIDATION,
        callbacks=[cb_checkpointer, cb_early_stopper])
print('hi2')        

for idx,layer in enumerate(model.layers):
        if layer.name == 'dense':
                print(idx,layer.name)
                #print(layer.get_weights())
                print("_____________________")


#model.load_weights("best.hdf5")

'''
print(fit_history.history.keys())

plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(fit_history.history['acc'])  
plt.plot(fit_history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()
'''
# NOTE that flow_from_directory treats each sub-folder as a class which works fine for training data
# Actually class_mode=None is a kind of workaround for test data which too must be kept in a subfolder

# batch_size can be 1 or any factor of test dataset size to ensure that test dataset is samples just once, i.e., no data is left out

test_generator = data_generator.flow_from_directory(
    directory = 'sketch/tx_000000000000',
    target_size = (image_size, image_size),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
)

#Need to compile layer[0] for extracting the 256- dim features.
#model.layers[1].compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

test_generator.reset()
#pred = model.layers[1].predict_generator(test_generator, steps = len(test_generator), verbose = 1) 
f =  Model(inputs=model.input,outputs=model.get_layer('dense').output)
f.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)
for idx,layer in enumerate(f.layers):
        if(layer.name == 'dense'):
                print(idx,layer.name)
                #print(layer.get_weights())
                print("________________")

pred = f.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
#Predicted labels
#pred2 = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
#predicted_class_indices = np.argmax(pred2, axis = 1)

fname = test_generator.filenames
sio.savemat('sketchmatnew.mat',mdict={'feature':pred,'label':fname})

print(pred.shape)

