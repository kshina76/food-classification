import numpy as np
import cv2
import os
import math
from glob import glob

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.models import Sequential, load_model, Model 
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from sklearn import preprocessing

class vgg16(object):

	def __init__(self, height, width, ch, num_classes):
		self.height = height
		self.width = width
		self.ch = ch
		self.num_classes = num_classes

	def build_model(self):
		inputs = Input((self.height, self.width, self.ch))
		x = Conv2D(64, (3, 3), padding='same', strides=1, activation='relu')(inputs)
		x = Conv2D(64, (3, 3), padding='same', strides=1, activation='relu')(x)
		x = MaxPooling2D((2, 2), strides=2,  padding='same')(x)
		x = Conv2D(128, (3, 3), padding='same', strides=1, activation='relu')(x)
		x = Conv2D(128, (3, 3), padding='same', strides=1, activation='relu')(x)
		x = MaxPooling2D((2, 2), strides=2,  padding='same')(x)
		x = Conv2D(256, (3, 3), padding='same', strides=1, activation='relu')(x)
		x = Conv2D(256, (3, 3), padding='same', strides=1, activation='relu')(x)
		x = Conv2D(256, (3, 3), padding='same', strides=1, activation='relu')(x)
		x = MaxPooling2D((2, 2), strides=2,  padding='same')(x)
		x = Conv2D(512, (3, 3), padding='same', strides=1, activation='relu')(x)
		x = Conv2D(512, (3, 3), padding='same', strides=1, activation='relu')(x)
		x = Conv2D(512, (3, 3), padding='same', strides=1, activation='relu')(x)
		x = MaxPooling2D((2, 2), strides=2,  padding='same')(x)
		x = Conv2D(512, (3, 3), padding='same', strides=1, activation='relu')(x)
		x = Conv2D(512, (3, 3), padding='same', strides=1, activation='relu')(x)
		x = Conv2D(512, (3, 3), padding='same', strides=1, activation='relu')(x)
		x = MaxPooling2D((2, 2), strides=2,  padding='same')(x)
		x = Flatten()(x)
		x = Dense(4096, name='dense1', activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(4096, name='dense2', activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(self.num_classes, activation='softmax')(x)

		model = Model(inputs=inputs, outputs=x)
		model.compile(loss='categorical_crossentropy',
					optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
					metrics=['accuracy'])
		model.summary()

		return model