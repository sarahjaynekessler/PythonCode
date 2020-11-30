import os
import keras
import tensorflow as tf 
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
"""
#galaxy stuff
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization
import h5py
import warnings
from keras import backend as K
"""

def makeAndRunModel(learningrate = None,epoch = None):

    if learningrate is None:
        lr = 1e-3
    else:
        lr = learningrate

    if epoch is None:
        eps = 30
    else:
        eps = epoch


    train_dir = '/Users/kessler.363/Desktop/S4G/Bar_Test'


    #model 4
    model = models.Sequential()

    #model.add(layers.Conv2D(32, (6, 6), activation='relu',
     #input_shape=(128, 128, 1)))
    model.add(layers.Conv2D(64, (3, 3), use_bias=False,
                        input_shape=(128, 128, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    #model.add(layers.Dropout(0.25))

    model.add(layers.MaxPooling2D((2, 2)))    

    model.add(layers.Conv2D(64, (3, 3), use_bias=False,)
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    
    model.add(layers.MaxPooling2D((2, 2)))    

    #model.add(layers.Dropout(0.25))

   
    model.add(layers.Conv2D(64, (3, 3), use_bias=False,)
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.25))


    model.add(layers.Conv2D(64, (3, 3), use_bias=False,)
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    
    model.add(layers.MaxPooling2D((2, 2)))  
    #model.add(layers.Dropout(0.25))


    model.add(layers.Flatten())
    
    model.add(layers.Dense(64, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Dense(64, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Dense(3, activation='softmax'))

    """

    #online galaxy class model
    
    num_classes=3
    input_shape=(128,128,1)
    model = Sequential()
    
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    """
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

    print(model.summary())
    train_datagen = ImageDataGenerator(
        validation_split = 0.1,
        rescale=1./255,
        rotation_range=360,
        width_shift_range=4,
        height_shift_range=4,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip = True
        )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        seed = 42,        
        shuffle=True,
        color_mode = 'grayscale',
        class_mode='categorical',
        subset = 'training')  

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        seed = 42,
        shuffle=True,
        color_mode = 'grayscale',        
        class_mode='categorical',
        subset = 'validation')

    history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples//train_generator.batch_size,
      epochs=eps,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples//validation_generator.batch_size)

    return(model,history)
